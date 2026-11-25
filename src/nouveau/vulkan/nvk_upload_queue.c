/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "nvk_upload_queue.h"

#include "nvk_device.h"
#include "nvk_physical_device.h"
#include "nvkmd/nvkmd.h"
#include "vk_alloc.h"

#include "nv_push.h"
#include "nv_push_cl9039.h"
#include "nv_push_cl90b5.h"
#include "cla097.h"

/* Before Kepler the copy class does nothing with these commands; uploads go
 * through M2MF there, as in the GL driver (nvc0_m2mf_copy_linear).
 */
static bool
nvk_upload_queue_uses_m2mf(struct nvk_device *dev)
{
   return nvk_device_physical(dev)->info.cls_eng3d < KEPLER_A;
}

static enum nvkmd_engines
nvk_upload_queue_engine(struct nvk_device *dev)
{
   return nvk_upload_queue_uses_m2mf(dev) ? NVKMD_ENGINE_M2MF
                                          : NVKMD_ENGINE_COPY;
}

static void
nvk_upload_queue_push_init(struct nvk_device *dev,
                           struct nvk_upload_queue *queue)
{
   nv_push_init(&queue->push, queue->push_data, ARRAY_SIZE(queue->push_data),
                nvk_queue_subchannels_from_engines(
                   nvk_upload_queue_engine(dev)));
}

/* One linear copy of at most 128 KiB */
static void
nvk_upload_queue_m2mf_copy(struct nv_push *p, uint64_t dst_addr,
                           uint64_t src_addr, uint32_t size)
{
   __push_mthd(p, SUBC_NV9039, 0x0238); /* OFFSET_OUT_HIGH, LOW */
   P_INLINE_DATA(p, dst_addr >> 32);
   P_INLINE_DATA(p, dst_addr & 0xffffffff);
   __push_mthd(p, SUBC_NV9039, 0x030c); /* OFFSET_IN_HIGH, LOW */
   P_INLINE_DATA(p, src_addr >> 32);
   P_INLINE_DATA(p, src_addr & 0xffffffff);
   __push_mthd(p, SUBC_NV9039, 0x031c); /* LINE_LENGTH_IN, LINE_COUNT */
   P_INLINE_DATA(p, size);
   P_INLINE_DATA(p, 1);
   __push_mthd(p, SUBC_NV9039, 0x0300); /* EXEC: linear in and out */
   P_INLINE_DATA(p, (1 << 20) | 0x10 | 0x100);
   /* The copy runs on after the method has been taken; whoever waits for
    * this queue's time point (a submit that uses a shader uploaded here)
    * must find the data in place.
    */
   __push_mthd(p, SUBC_NV9039, 0x0110); /* WAIT_FOR_IDLE */
   P_INLINE_DATA(p, 0);
}

VkResult
nvk_upload_queue_init(struct nvk_device *dev,
                      struct nvk_upload_queue *queue)
{
   VkResult result;

   memset(queue, 0, sizeof(*queue));

   simple_mtx_init(&queue->mutex, mtx_plain);

   result = nvkmd_dev_create_ctx(dev->nvkmd, &dev->vk.base,
                                 nvk_upload_queue_engine(dev), &queue->ctx);
   if (result != VK_SUCCESS)
      goto fail_mutex;

   result = nvk_mem_stream_init(dev, &queue->stream);
   if (result != VK_SUCCESS)
      goto fail_ctx;

   nvk_upload_queue_push_init(dev, queue);

   if (nvk_upload_queue_uses_m2mf(dev)) {
      /* the subchannel has to be told its object, once */
      struct nv_push *p = &queue->push;
      P_MTHD(p, NV9039, SET_OBJECT);
      P_NV9039_SET_OBJECT(p, {
         .class_id = nvk_device_physical(dev)->info.cls_m2mf,
         .engine_id = 0,
      });
   }

   return VK_SUCCESS;

fail_ctx:
   nvkmd_ctx_destroy(queue->ctx);
fail_mutex:
   simple_mtx_destroy(&queue->mutex);

   return result;
}

void
nvk_upload_queue_finish(struct nvk_device *dev,
                        struct nvk_upload_queue *queue)
{
   nvk_mem_stream_finish(dev, &queue->stream);
   nvkmd_ctx_destroy(queue->ctx);
   simple_mtx_destroy(&queue->mutex);
}

static VkResult
nvk_upload_queue_flush_locked(struct nvk_device *dev,
                              struct nvk_upload_queue *queue,
                              uint64_t *time_point_out)
{
   VkResult result;

   if (time_point_out != NULL)
      *time_point_out = queue->last_time_point;

   if (nv_push_dw_count(&queue->push) == 0)
      return VK_SUCCESS;

   /* nvk_mem_stream_flush() will not update last_time_point if it fails.  If
    * we do fail and lose the device, nvk_upload_queue_sync won't wait forever
    * on a time point that will never signal.
    */
   result = nvk_mem_stream_push(dev, &queue->stream, queue->ctx,
                                queue->push_data,
                                nv_push_dw_count(&queue->push),
                                &queue->last_time_point);
   if (result != VK_SUCCESS)
      return result;

   nvk_upload_queue_push_init(dev, queue);

   if (time_point_out != NULL)
      *time_point_out = queue->last_time_point;

   return VK_SUCCESS;
}

VkResult
nvk_upload_queue_flush(struct nvk_device *dev,
                       struct nvk_upload_queue *queue,
                       uint64_t *time_point_out)
{
   VkResult result;

   simple_mtx_lock(&queue->mutex);
   result = nvk_upload_queue_flush_locked(dev, queue, time_point_out);
   simple_mtx_unlock(&queue->mutex);

   return result;
}

static VkResult
nvk_upload_queue_sync_locked(struct nvk_device *dev,
                             struct nvk_upload_queue *queue)
{
   VkResult result;

   result = nvk_upload_queue_flush_locked(dev, queue, NULL);
   if (result != VK_SUCCESS)
      return result;

   if (queue->last_time_point == 0)
      return VK_SUCCESS;

   return vk_sync_wait(&dev->vk, queue->stream.sync, queue->last_time_point,
                       VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
}

VkResult
nvk_upload_queue_sync(struct nvk_device *dev,
                      struct nvk_upload_queue *queue)
{
   VkResult result;

   simple_mtx_lock(&queue->mutex);
   result = nvk_upload_queue_sync_locked(dev, queue);
   simple_mtx_unlock(&queue->mutex);

   return result;
}

static VkResult
nvk_upload_queue_upload_locked(struct nvk_device *dev,
                               struct nvk_upload_queue *queue,
                               uint64_t dst_addr,
                               const void *src, size_t size)
{
   VkResult result;

   assert(dst_addr % 4 == 0);
   assert(size % 4 == 0);

   while (size > 0) {
      const uint32_t cmd_size_dw = 14;
      if (queue->push.end + cmd_size_dw > queue->push.limit) {
         result = nvk_upload_queue_flush_locked(dev, queue, NULL);
         if (result != VK_SUCCESS)
            return result;
      }
      struct nv_push *p = &queue->push;

      const uint32_t data_size = MIN2(size, NVK_MEM_STREAM_MAX_ALLOC_SIZE);

      uint64_t data_addr;
      void *data_map;
      result = nvk_mem_stream_alloc(dev, &queue->stream, data_size, 4,
                                    &data_addr, &data_map);
      if (result != VK_SUCCESS)
         return result;

      memcpy(data_map, src, data_size);

      assert(data_size <= (1 << 17));

      if (nvk_upload_queue_uses_m2mf(dev)) {
         nvk_upload_queue_m2mf_copy(p, dst_addr, data_addr, data_size);

         dst_addr += data_size;
         src += data_size;
         size -= data_size;
         continue;
      }

      P_MTHD(p, NV90B5, OFFSET_IN_UPPER);
      P_NV90B5_OFFSET_IN_UPPER(p, data_addr >> 32);
      P_NV90B5_OFFSET_IN_LOWER(p, data_addr & 0xffffffff);
      P_NV90B5_OFFSET_OUT_UPPER(p, dst_addr >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, dst_addr & 0xffffffff);
      P_NV90B5_PITCH_IN(p, data_size);
      P_NV90B5_PITCH_OUT(p, data_size);
      P_NV90B5_LINE_LENGTH_IN(p, data_size);
      P_NV90B5_LINE_COUNT(p, 1);

      P_IMMD(p, NV90B5, LAUNCH_DMA, {
         .data_transfer_type = DATA_TRANSFER_TYPE_NON_PIPELINED,
         .multi_line_enable = MULTI_LINE_ENABLE_FALSE,
         .flush_enable = FLUSH_ENABLE_TRUE,
         .src_memory_layout = SRC_MEMORY_LAYOUT_PITCH,
         .dst_memory_layout = DST_MEMORY_LAYOUT_PITCH,
      });

      dst_addr += data_size;
      src += data_size;
      size -= data_size;
   }

   return VK_SUCCESS;
}

VkResult
nvk_upload_queue_upload(struct nvk_device *dev,
                        struct nvk_upload_queue *queue,
                        uint64_t dst_addr,
                        const void *src, size_t size)
{
   VkResult result;

   simple_mtx_lock(&queue->mutex);
   result = nvk_upload_queue_upload_locked(dev, queue, dst_addr, src, size);
   simple_mtx_unlock(&queue->mutex);

   return result;
}

static VkResult
nvk_upload_queue_fill_locked(struct nvk_device *dev,
                             struct nvk_upload_queue *queue,
                             uint64_t dst_addr, uint32_t data, size_t size)
{
   VkResult result;

   assert(dst_addr % 4 == 0);
   assert(size % 4 == 0);

   while (size > 0) {
      const uint32_t cmd_size_dw = 16;
      if (queue->push.end + cmd_size_dw > queue->push.limit) {
         result = nvk_upload_queue_flush_locked(dev, queue, NULL);
         if (result != VK_SUCCESS)
            return result;
      }
      struct nv_push *p = &queue->push;

      if (nvk_upload_queue_uses_m2mf(dev)) {
         /* M2MF has no constant source: copy from a block of the value */
         const uint32_t fill_size = MIN2(size, NVK_MEM_STREAM_MAX_ALLOC_SIZE);

         uint64_t fill_addr;
         void *fill_map;
         result = nvk_mem_stream_alloc(dev, &queue->stream, fill_size, 4,
                                       &fill_addr, &fill_map);
         if (result != VK_SUCCESS)
            return result;

         for (uint32_t i = 0; i < fill_size / 4; i++)
            ((uint32_t *)fill_map)[i] = data;

         nvk_upload_queue_m2mf_copy(p, dst_addr, fill_addr, fill_size);

         dst_addr += fill_size;
         size -= fill_size;
         continue;
      }

      const uint32_t max_dim = 1 << 17;
      uint32_t width_B, height;
      if (size > max_dim) {
         width_B = max_dim;
         height = MIN2(max_dim, size / width_B);
      } else {
         width_B = size;
         height = 1;
      }
      assert(width_B * height <= size);

      P_MTHD(p, NV90B5, OFFSET_OUT_UPPER);
      P_NV90B5_OFFSET_OUT_UPPER(p, dst_addr >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, dst_addr & 0xffffffff);
      P_NV90B5_PITCH_IN(p, width_B);
      P_NV90B5_PITCH_OUT(p, width_B);
      P_NV90B5_LINE_LENGTH_IN(p, width_B / 4);
      P_NV90B5_LINE_COUNT(p, height);

      P_IMMD(p, NV90B5, SET_REMAP_CONST_A, data);
      P_IMMD(p, NV90B5, SET_REMAP_COMPONENTS, {
         .dst_x = DST_X_CONST_A,
         .dst_y = DST_Y_CONST_A,
         .dst_z = DST_Z_CONST_A,
         .dst_w = DST_W_CONST_A,
         .component_size = COMPONENT_SIZE_FOUR,
         .num_src_components = NUM_SRC_COMPONENTS_ONE,
         .num_dst_components = NUM_DST_COMPONENTS_ONE,
      });

      P_IMMD(p, NV90B5, LAUNCH_DMA, {
         .data_transfer_type = DATA_TRANSFER_TYPE_NON_PIPELINED,
         .multi_line_enable = height > 1,
         .flush_enable = FLUSH_ENABLE_TRUE,
         .src_memory_layout = SRC_MEMORY_LAYOUT_PITCH,
         .dst_memory_layout = DST_MEMORY_LAYOUT_PITCH,
         .remap_enable = REMAP_ENABLE_TRUE,
      });

      dst_addr += width_B * height;
      size -= width_B * height;
   }

   return VK_SUCCESS;
}

VkResult
nvk_upload_queue_fill(struct nvk_device *dev,
                      struct nvk_upload_queue *queue,
                      uint64_t dst_addr, uint32_t data, size_t size)
{
   VkResult result;

   simple_mtx_lock(&queue->mutex);
   result = nvk_upload_queue_fill_locked(dev, queue, dst_addr, data, size);
   simple_mtx_unlock(&queue->mutex);

   return result;
}
