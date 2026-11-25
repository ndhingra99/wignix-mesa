/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "nvk_buffer.h"
#include "nvk_cmd_buffer.h"
#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_mme.h"
#include "nvk_physical_device.h"
#include "nvk_shader.h"

#include "cl906f.h"
#include "cla0b5.h"
#include "cla1c0.h"
#include "clc0c0.h"
#include "clc5c0.h"
#include "clcbc0.h"
#include "clcdc0.h"
#include "nv_push_cl90c0.h"
#include "nv_push_cl9097.h"
#include "nv_push_cla0c0.h"
#include "nv_push_clb0c0.h"
#include "nv_push_clb1c0.h"
#include "nv_push_clc3c0.h"
#include "nv_push_clc597.h"
#include "nv_push_clc6c0.h"
#include "nv_push_clc7c0.h"
#include "nv_push_clc86f.h"

/* Called on Fermi's compute class: launches a grid whose three group counts
 * are its parameters, so that they can come straight from an indirect
 * buffer.  A zero count launches nothing.  The macro memory is one for the
 * 3D and the compute class, so this lives among the 3D macros.
 */
void
nvk_mme_fermi_launch_grid(struct mme_builder *b)
{
   if (b->devinfo->cls_compute >= KEPLER_COMPUTE_A)
      return;

   struct mme_value x = mme_load(b);
   struct mme_value y = mme_load(b);
   struct mme_value z = mme_load(b);

   mme_if(b, ine, x, mme_zero()) {
      mme_if(b, ine, y, mme_zero()) {
         mme_if(b, ine, z, mme_zero()) {
            struct mme_value yx = mme_merge(b, x, y, 16, 16, 0);
            mme_mthd(b, NV90C0_SET_CTA_RASTER_SIZE_A);
            mme_emit(b, yx);
            mme_emit(b, z);
            mme_mthd(b, NV90C0_BEGIN_GRID);
            mme_emit(b, mme_zero());
            mme_mthd(b, 0x0a08);
            mme_emit(b, mme_zero());
            mme_mthd(b, NV90C0_LAUNCH);
            mme_emit(b, mme_imm(0x1000));
            mme_mthd(b, 0x0a04); /* end of grid */
            mme_emit(b, mme_zero());
            mme_mthd(b, NV90C0_WAIT_REF_COUNT);
            mme_emit(b, mme_imm(1));
         }
      }
   }
}

VkResult
nvk_push_dispatch_state_init(struct nvk_queue *queue, struct nv_push *p)
{
   struct nvk_device *dev = nvk_queue_device(queue);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);

   P_MTHD(p, NV90C0, SET_OBJECT);
   P_NV90C0_SET_OBJECT(p, {
      .class_id = pdev->info.cls_compute,
      .engine_id = 0,
   });

   if (pdev->info.cls_compute == MAXWELL_COMPUTE_A)
      P_IMMD(p, NVB0C0, SET_SELECT_MAXWELL_TEXTURE_HEADERS, V_TRUE);

   if (pdev->info.cls_compute < KEPLER_COMPUTE_A) {
      /* Fermi's compute engine is set up by methods, as the GL driver does
       * it (nvc0_screen_compute_setup): the number of processors, the call
       * depth, the map from global addresses to virtual ones (identity for
       * all 256 values of the top byte, read and write), the cache split.
       */
      __push_immd(p, SUBC_NV90C0, 0x0758, pdev->info.tpc_count);
      __push_immd(p, SUBC_NV90C0, 0x0d64, 0xf);
      __push_mthd(p, SUBC_NV90C0, NV90C0_SET_WORK_DISTRIBUTION);
      P_INLINE_DATA(p, 0x8000);

      P_IMMD(p, NV90C0, SET_GA_TO_VA_MAPPING_MODE, V_DISABLE);
      P_0INC(p, NV90C0, LOAD_GA_TO_VA_MAPPING_ENTRY);
      for (uint32_t i = 0; i <= 0xff; i++)
         P_INLINE_DATA(p, (0xc << 28) | (i << 16) | i);
      P_IMMD(p, NV90C0, SET_GA_TO_VA_MAPPING_MODE, V_ENABLE);

      __push_immd(p, SUBC_NV90C0, NV90C0_SET_L1_CONFIGURATION,
         NV90C0_SET_L1_CONFIGURATION_DIRECTLY_ADDRESSABLE_MEMORY_SIZE_48KB);
      __push_immd(p, SUBC_NV90C0, NV90C0_SET_SHADER_SHARED_MEMORY_SIZE, 0);

      /* How many bound units the shaders can reach: 16 samplers and 32
       * texture headers (NVK_MAX_SAMPLER_UNITS, NVK_MAX_TEX_UNITS), the GL
       * driver's TEX_LIMITS value.  Without it units from 4 on read as
       * unbound.
       */
      P_IMMD(p, NV90C0, SET_BINDING_CONTROL_TEXTURE, {
         .max_active_samplers = MAX_ACTIVE_SAMPLERS__16,
         .max_active_headers = MAX_ACTIVE_HEADERS__32,
      });
   }

   if (pdev->info.cls_compute < VOLTA_COMPUTE_A) {
      uint64_t shader_base_addr =
         nvk_heap_contiguous_base_address(&dev->shader_heap);

      P_MTHD(p, NVA0C0, SET_PROGRAM_REGION_A);
      P_NVA0C0_SET_PROGRAM_REGION_A(p, shader_base_addr >> 32);
      P_NVA0C0_SET_PROGRAM_REGION_B(p, shader_base_addr);
   }

   if (pdev->info.cls_compute >= VOLTA_COMPUTE_A) {
      /* From nvc0_screen.c:
       *
       *    "Reduce likelihood of collision with real buffers by placing the
       *    hole at the top of the 4G area. This will have to be dealt with
       *    for real eventually by blocking off that area from the VM."
       *
       * Really?!?  TODO: Fix this for realz.
       */
      uint64_t temp;

      /* shared memory window needs 4GB alignment on hopper+ */
      if (pdev->info.cls_compute >= HOPPER_COMPUTE_A)
         temp = 1ULL << 32;
      else
         temp = 0xfeULL << 24;

      P_MTHD(p, NVC3C0, SET_SHADER_SHARED_MEMORY_WINDOW_A);
      P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_A(p, temp >> 32);
      P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_B(p, temp & 0xffffffff);

      temp = 0xffULL << 24;
      P_MTHD(p, NVC3C0, SET_SHADER_LOCAL_MEMORY_WINDOW_A);
      P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A(p, temp >> 32);
      P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B(p, temp & 0xffffffff);
   } else {
      P_MTHD(p, NVA0C0, SET_SHADER_LOCAL_MEMORY_WINDOW);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_WINDOW(p, 0xff << 24);

      P_MTHD(p, NVA0C0, SET_SHADER_SHARED_MEMORY_WINDOW);
      P_NVA0C0_SET_SHADER_SHARED_MEMORY_WINDOW(p, 0xfe << 24);
   }

   return VK_SUCCESS;
}

static inline uint16_t
nvk_cmd_buffer_compute_cls(struct nvk_cmd_buffer *cmd)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);
   return pdev->info.cls_compute;
}

void
nvk_cmd_buffer_begin_compute(struct nvk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   if (cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
      struct nv_push *p = nvk_cmd_buffer_push(cmd, 6);
      if (nvk_cmd_buffer_compute_cls(cmd) >= MAXWELL_COMPUTE_B) {
         P_IMMD(p, NVB1C0, INVALIDATE_SKED_CACHES, 0);
      }
      P_IMMD(p, NVA0C0, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
         .lines = LINES_ALL,
      });
      P_IMMD(p, NVA0C0, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
         .lines = LINES_ALL,
      });
   }
}

void
nvk_cmd_invalidate_compute_state(struct nvk_cmd_buffer *cmd)
{
   memset(&cmd->state.cs, 0, sizeof(cmd->state.cs));
}

void
nvk_cmd_bind_compute_shader(struct nvk_cmd_buffer *cmd,
                            struct nvk_shader *shader)
{
   cmd->state.cs.shader = shader;
}

static uint32_t
nvk_compute_local_size(struct nvk_cmd_buffer *cmd)
{
   const struct nvk_shader *shader = cmd->state.cs.shader;

   return shader->info.cs.local_size[0] *
          shader->info.cs.local_size[1] *
          shader->info.cs.local_size[2];
}

static void
nvk_flush_compute_state(struct nvk_cmd_buffer *cmd,
                        uint32_t base_workgroup[3],
                        uint32_t global_size[3])
{
   struct nvk_descriptor_state *desc = &cmd->state.cs.descriptors;

   nvk_cmd_buffer_flush_push_descriptors(cmd, desc);

   nvk_descriptor_state_set_root_array(cmd, desc, cs.base_group,
                                       0, 3, base_workgroup);
   nvk_descriptor_state_set_root_array(cmd, desc, cs.group_count,
                                       0, 3, global_size);

   if (NAK_CAN_PRINTF)
      nvk_cmd_buffer_flush_printf_buffer(cmd, desc);
}

static VkResult
nvk_cmd_upload_qmd(struct nvk_cmd_buffer *cmd,
                   const struct nvk_shader *shader,
                   const struct nvk_descriptor_state *desc,
                   const void *root,
                   size_t root_size,
                   uint32_t global_size[3],
                   uint64_t *qmd_addr_out,
                   uint64_t *root_desc_addr_out)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);
   const uint32_t min_cbuf_alignment = nvk_min_cbuf_alignment(&pdev->info);
   const uint32_t qmd_size_B = nak_qmd_size_B(&pdev->info);
   VkResult result;

   /* pre Pascal the constant buffer sizes need to be 0x100 aligned. As we
    * simply allocated a buffer and upload data to it, make sure its size is
    * 0x100 aligned.
    */
   assert(root_size % min_cbuf_alignment == 0);

   void *root_desc_map;
   uint64_t root_desc_addr;
   result = nvk_cmd_buffer_upload_alloc(cmd, root_size, min_cbuf_alignment,
                                        &root_desc_addr, &root_desc_map);
   if (unlikely(result != VK_SUCCESS))
      return result;

   memcpy(root_desc_map, root, root_size);

   uint64_t qmd_addr = 0;
   if (shader != NULL) {
      struct nak_qmd_info qmd_info = {
         .addr = shader->hdr_addr,
         .smem_size = shader->info.cs.smem_size,
         .global_size = {
            global_size[0],
            global_size[1],
            global_size[2],
         },
      };

      assert(shader->cbuf_map.cbuf_count <= ARRAY_SIZE(qmd_info.cbufs));
      for (uint32_t c = 0; c < shader->cbuf_map.cbuf_count; c++) {
         const struct nvk_cbuf *cbuf = &shader->cbuf_map.cbufs[c];

         struct nvk_buffer_address ba;
         if (cbuf->type == NVK_CBUF_TYPE_ROOT_DESC) {
            ba = (struct nvk_buffer_address) {
               .base_addr = root_desc_addr,
               .size = root_size,
            };
         } else {
            ASSERTED bool direct_descriptor =
               nvk_cmd_buffer_get_cbuf_addr(cmd, desc, shader, cbuf, &ba);
            assert(direct_descriptor);
         }

         if (ba.size > 0) {
            assert(ba.base_addr % min_cbuf_alignment == 0);
            ba.size = align(ba.size, min_cbuf_alignment);
            ba.size = MIN2(ba.size, NVK_MAX_CBUF_SIZE);

            qmd_info.cbufs[qmd_info.num_cbufs++] = (struct nak_qmd_cbuf) {
               .index = c,
               .addr = ba.base_addr,
               .size = ba.size,
            };
         }
      }

      uint32_t qmd[NAK_MAX_QMD_DWORDS];
      assert(qmd_size_B <= sizeof(qmd));
      nak_fill_qmd(&pdev->info, &shader->info, &qmd_info, qmd, qmd_size_B);

      void *qmd_map;
      result = nvk_cmd_buffer_alloc_qmd(cmd, qmd_size_B, 0x100,
                                        &qmd_addr, &qmd_map);
      if (unlikely(result != VK_SUCCESS))
         return result;

      memcpy(qmd_map, qmd, qmd_size_B);
   }

   *qmd_addr_out = qmd_addr;
   if (root_desc_addr_out != NULL)
      *root_desc_addr_out = root_desc_addr;

   return VK_SUCCESS;
}

VkResult
nvk_cmd_flush_cs_qmd(struct nvk_cmd_buffer *cmd,
                     const struct nvk_cmd_state *state,
                     uint32_t global_size[3],
                     uint64_t *qmd_addr_out,
                     uint64_t *root_desc_addr_out)
{
   const struct nvk_descriptor_state *desc = &state->cs.descriptors;
   STATIC_ASSERT((sizeof(desc->root) & 0xff) == 0);

   return nvk_cmd_upload_qmd(cmd, state->cs.shader, desc,
                             desc->root, sizeof(desc->root),
                             global_size,
                             qmd_addr_out, root_desc_addr_out);
}

/* A dispatch on Fermi (compute class 90c0).  There is no launch descriptor:
 * the constant buffers, the texture and sampler units, the program and the
 * sizes are set by methods and LAUNCH starts the grid, following the GL
 * driver (nvc0_launch_grid).  Constant buffer and texture bindings of the
 * compute engine are shared with the 3D engine's, so the 3D ones are marked
 * for rebinding.
 */
static void
nvk_cmd_dispatch_fermi(struct nvk_cmd_buffer *cmd,
                       const struct nvk_shader *shader,
                       const struct nvk_descriptor_state *desc,
                       const void *root, size_t root_size,
                       const uint32_t group_count[3],
                       bool indirect, uint64_t indirect_addr)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);

   if (shader == NULL)
      return;

   if (!indirect &&
       (group_count[0] == 0 || group_count[1] == 0 || group_count[2] == 0))
      return;

   /* The engine wants a local memory area even for a shader that uses
    * none: every launch declares a call stack of 0x800 bytes per warp.
    */
   nvk_device_ensure_slm(dev, shader->info.slm_size,
                         MAX2(shader->info.crs_size, 0x800) * 4);

   /* Fermi compute shaders use no constant buffers (nvk_lower_nir): the
    * compute root table has one fixed place in memory, written here before
    * every launch and read by address.
    */
   assert(root_size <= dev->fermi_cs_root->size_B);
   assert(root_size % 4 == 0);
   const uint64_t root_desc_addr = dev->fermi_cs_root->va->addr;
   const uint32_t *root_dw = root;

   const struct nvk_cbuf_map *map = &shader->cbuf_map;
   struct nv_push *p;

   /* Written through the 3D class, as the indirect dispatch macro writes
    * its counts.  Any constant buffer method of the compute class (select,
    * load or bind) makes 3D shaders read the compute buffer in place of
    * their own root table afterwards (tests/vkdawn.c Z9).
    */
   p = nvk_cmd_buffer_push(cmd, 4);
   P_MTHD(p, NV9097, SET_CONSTANT_BUFFER_SELECTOR_A);
   P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_A(p, root_size);
   P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_B(p, root_desc_addr >> 32);
   P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_C(p, root_desc_addr);
   for (uint32_t dw = 0; dw < root_size / 4; dw += 256) {
      const uint32_t len_dw = MIN2(root_size / 4 - dw, 256);
      p = nvk_cmd_buffer_push(cmd, 2 + len_dw);
      P_1INC(p, NV9097, LOAD_CONSTANT_BUFFER_OFFSET);
      P_NV9097_LOAD_CONSTANT_BUFFER_OFFSET(p, dw * 4);
      P_INLINE_ARRAY(p, &root_dw[dw], len_dw);
   }

   if (indirect) {
      /* the 3D engine's macro writes the counts into the root table and
       * counts the invocations
       */
      p = nvk_cmd_buffer_push(cmd, 5);
      /* Stall the command streamer */
      __push_immd(p, SUBC_NV9097, NV906F_SET_REFERENCE, 0);
      P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_DISPATCH_INDIRECT));
      nv_push_update_count(p, sizeof(VkDispatchIndirectCommand) / 4);
      nvk_cmd_buffer_push_indirect(cmd, indirect_addr,
                                   sizeof(VkDispatchIndirectCommand));
      p = nvk_cmd_buffer_push(cmd, 3);
      P_INLINE_DATA(p, root_desc_addr >> 32);
      P_INLINE_DATA(p, root_desc_addr);
      P_INLINE_DATA(p, shader->info.cs.local_size[0] *
                       shader->info.cs.local_size[1] *
                       shader->info.cs.local_size[2]);
   }

   /* no constant buffers: the shader reads the root table and everything
    * it points to from memory (nvk_lower_nir)
    */
   assert(map->cbuf_count == 0);

   if (map->tex_unit_count > 0 || map->sampler_unit_count > 0) {
      p = nvk_cmd_buffer_push(cmd, 2 * (map->tex_unit_count +
                                        map->sampler_unit_count));
      for (uint32_t i = 0; i < map->tex_unit_count; i++) {
         const struct nvk_sampled_image_descriptor *d = desc == NULL ? NULL :
            nvk_cmd_buffer_get_tex_unit_desc(desc, &map->tex_units[i]);

         P_IMMD(p, NV90C0, BIND_TEXTURE_HEADER, {
            .valid = d != NULL,
            .texture_slot = i,
            .index = d != NULL ? d->image_index : 0,
         });
      }
      for (uint32_t i = 0; i < map->sampler_unit_count; i++) {
         const struct nvk_tex_unit *unit = &map->sampler_units[i];
         const struct nvk_sampled_image_descriptor *d = desc == NULL ? NULL :
            nvk_cmd_buffer_get_tex_unit_desc(desc, unit);

         P_IMMD(p, NV90C0, BIND_TEXTURE_SAMPLER, {
            .valid = d != NULL || unit->type == NVK_TEX_UNIT_TYPE_TXF_SAMPLER,
            .sampler_slot = i,
            .index = d != NULL ? d->sampler_index : 0,
         });
      }
   }

   const uint32_t threads = shader->info.cs.local_size[0] *
                            shader->info.cs.local_size[1] *
                            shader->info.cs.local_size[2];

   p = nvk_cmd_buffer_push(cmd, 42);

   /* Internal shaders (no descriptor state) don't want conditional
    * rendering
    */
   if (desc == NULL)
      P_IMMD(p, NVA0C0, SET_RENDER_ENABLE_OVERRIDE, MODE_ALWAYS_RENDER);

   /* the engine keeps its own copy of the code it has run */
   __push_mthd(p, SUBC_NV90C0, 0x1698); /* FLUSH: code */
   P_INLINE_DATA(p, 0x1);
   P_MTHD(p, NV90C0, SET_CTA_PROGRAM_START);
   P_INLINE_DATA(p, shader->hdr_addr);

   P_MTHD(p, NV90C0, SET_SHADER_LOCAL_MEMORY_LOW_SIZE);
   P_INLINE_DATA(p, align(shader->info.slm_size, 0x10));
   P_INLINE_DATA(p, 0);
   P_INLINE_DATA(p, 0x800);

   P_MTHD(p, NV90C0, SET_SHADER_SHARED_MEMORY_SIZE);
   P_INLINE_DATA(p, align(shader->info.cs.smem_size, 0x100));
   P_INLINE_DATA(p, threads);
   P_INLINE_DATA(p, shader->info.num_control_barriers);
   P_MTHD(p, NV90C0, SET_CTA_REGISTER_COUNT);
   P_INLINE_DATA(p, shader->info.num_gprs);

   __push_mthd(p, SUBC_NV90C0, 0x0780); /* grid id */
   P_INLINE_DATA(p, 1);
   P_MTHD(p, NV90C0, SET_LAUNCH_ID);
   P_INLINE_DATA(p, 0);
   __push_mthd(p, SUBC_NV90C0, 0x1698); /* FLUSH: global memory */
   P_INLINE_DATA(p, 0x110);

   P_MTHD(p, NV90C0, SET_CTA_THREAD_DIMENSION_A);
   P_INLINE_DATA(p, (shader->info.cs.local_size[1] << 16) |
                    shader->info.cs.local_size[0]);
   P_INLINE_DATA(p, shader->info.cs.local_size[2]);

   if (indirect) {
      /* the compute engine's macro launches with the counts as they are
       * in the buffer
       */
      __push_immd(p, SUBC_NV90C0, NV906F_SET_REFERENCE, 0);
      P_1INC(p, NV90C0, CALL_MME_MACRO(NVK_MME_FERMI_LAUNCH_GRID));
      nv_push_update_count(p, sizeof(VkDispatchIndirectCommand) / 4);
      nvk_cmd_buffer_push_indirect(cmd, indirect_addr,
                                   sizeof(VkDispatchIndirectCommand));
      p = nvk_cmd_buffer_push(cmd, 2);
   } else {
      P_MTHD(p, NV90C0, SET_CTA_RASTER_SIZE_A);
      P_INLINE_DATA(p, (group_count[1] << 16) | group_count[0]);
      P_INLINE_DATA(p, group_count[2]);

      P_MTHD(p, NV90C0, BEGIN_GRID);
      P_INLINE_DATA(p, 0);
      __push_mthd(p, SUBC_NV90C0, 0x0a08);
      P_INLINE_DATA(p, 0);
      P_MTHD(p, NV90C0, LAUNCH);
      P_INLINE_DATA(p, 0x1000);
      __push_mthd(p, SUBC_NV90C0, 0x0a04); /* end of grid */
      P_INLINE_DATA(p, 0);
      P_MTHD(p, NV90C0, WAIT_REF_COUNT);
      P_INLINE_DATA(p, 0x1);
   }

   if (desc == NULL)
      P_IMMD(p, NVA0C0, SET_RENDER_ENABLE_OVERRIDE, MODE_USE_RENDER_ENABLE);

   for (uint32_t g = 0; g < ARRAY_SIZE(cmd->state.gfx.cbuf_groups); g++) {
      struct nvk_cbuf_group *group = &cmd->state.gfx.cbuf_groups[g];
      group->dirty |= BITFIELD_MASK(ARRAY_SIZE(group->cbufs));
   }

   /* The constant buffer selector is shared too, and root descriptor and
    * draw parameter writes go through it into the 3D root table: point it
    * back there, with the method filter off so that the write is not
    * dropped for repeating the 3D class's last value.  The next draw binds
    * the 3D constant buffers and texture units again the same way
    * (nvk_cmd_flush_gfx_state).
    */
   p = nvk_cmd_buffer_push(cmd, 6);
   P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL, MODE_METHOD_TRACK);
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SELECT_CB0));
   P_INLINE_DATA(p, 0);
   P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL,
          MODE_METHOD_TRACK_WITH_FILTER);
   cmd->state.gfx.fermi_rebind = true;
}

static void
nvk_build_mme_add_cs_invocations(struct mme_builder *b,
                                 struct mme_value64 count)
{
   struct mme_value accum_hi = nvk_mme_load_scratch(b, CS_INVOCATIONS_HI);
   struct mme_value accum_lo = nvk_mme_load_scratch(b, CS_INVOCATIONS_LO);
   struct mme_value64 accum = mme_value64(accum_lo, accum_hi);

   mme_add64_to(b, accum, accum, count);

   STATIC_ASSERT(NVK_MME_SCRATCH_CS_INVOCATIONS_HI + 1 ==
                 NVK_MME_SCRATCH_CS_INVOCATIONS_LO);

   mme_mthd(b, NVK_SET_MME_SCRATCH(CS_INVOCATIONS_HI));
   mme_emit(b, accum.hi);
   mme_emit(b, accum.lo);

   mme_free_reg64(b, accum);
}

void
nvk_mme_add_cs_invocations(struct mme_builder *b)
{
   struct mme_value64 count = mme_load_addr64(b);

   nvk_build_mme_add_cs_invocations(b, count);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdDispatchBase(VkCommandBuffer commandBuffer,
                    uint32_t baseGroupX,
                    uint32_t baseGroupY,
                    uint32_t baseGroupZ,
                    uint32_t groupCountX,
                    uint32_t groupCountY,
                    uint32_t groupCountZ)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);

   uint32_t base_workgroup[3] = { baseGroupX, baseGroupY, baseGroupZ };
   uint32_t global_size[3] = { groupCountX, groupCountY, groupCountZ };
   nvk_flush_compute_state(cmd, base_workgroup, global_size);

   if (nvk_cmd_buffer_compute_cls(cmd) < KEPLER_COMPUTE_A) {
      const struct nvk_descriptor_state *desc = &cmd->state.cs.descriptors;
      nvk_cmd_dispatch_fermi(cmd, cmd->state.cs.shader, desc,
                             desc->root, sizeof(desc->root), global_size,
                             false, 0);

      /* Only counted if we have a compute shader invocations query */
      if (cmd->state.cs.active_compute_invocations_query ||
          cmd->state.inherited_pipeline_statistics &
             VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT) {
         const uint64_t cs_invocations =
            (uint64_t)nvk_compute_local_size(cmd) * (uint64_t)groupCountX *
            (uint64_t)groupCountY * (uint64_t)groupCountZ;

         struct nv_push *p = nvk_cmd_buffer_push(cmd, 3);
         P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_ADD_CS_INVOCATIONS));
         P_INLINE_DATA(p, cs_invocations >> 32);
         P_INLINE_DATA(p, cs_invocations);
      }
      return;
   }

   uint64_t qmd_addr = 0;
   VkResult result = nvk_cmd_flush_cs_qmd(cmd, &cmd->state, global_size,
                                          &qmd_addr, NULL);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   /* Only emit this if we have a compute shader invocations query */
   if (cmd->state.cs.active_compute_invocations_query ||
       cmd->state.inherited_pipeline_statistics &
          VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT) {
      const uint32_t local_size = nvk_compute_local_size(cmd);
      const uint64_t cs_invocations =
         (uint64_t)local_size * (uint64_t)groupCountX *
         (uint64_t)groupCountY * (uint64_t)groupCountZ;

      struct nv_push *p = nvk_cmd_buffer_push(cmd, 3);
      if (nvk_cmd_buffer_compute_cls(cmd) >= AMPERE_COMPUTE_B)
         P_1INC(p, NVC7C0, CALL_MME_MACRO(NVK_MME_ADD_CS_INVOCATIONS));
      else
         P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_ADD_CS_INVOCATIONS));
      P_INLINE_DATA(p, cs_invocations >> 32);
      P_INLINE_DATA(p, cs_invocations);
   }

   struct nv_push *p = nvk_cmd_buffer_push(cmd, 4);
   P_MTHD(p, NVA0C0, SEND_PCAS_A);
   P_NVA0C0_SEND_PCAS_A(p, qmd_addr >> 8);

   if (nvk_cmd_buffer_compute_cls(cmd) <= TURING_COMPUTE_A) {
      P_IMMD(p, NVA0C0, SEND_SIGNALING_PCAS_B, {
            .invalidate = INVALIDATE_TRUE,
            .schedule = SCHEDULE_TRUE
      });
   } else {
      P_IMMD(p, NVC6C0, SEND_SIGNALING_PCAS2_B,
             PCAS_ACTION_INVALIDATE_COPY_SCHEDULE);
   }
}

void
nvk_cmd_dispatch_with_root(struct nvk_cmd_buffer *cmd,
                           struct nvk_shader *shader,
                           const void *root,
                           size_t root_size,
                           uint32_t groupCountX,
                           uint32_t groupCountY,
                           uint32_t groupCountZ)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   uint32_t group_count[3] = { groupCountX, groupCountY, groupCountZ };

   if (nvk_cmd_buffer_compute_cls(cmd) < KEPLER_COMPUTE_A) {
      nvk_cmd_dispatch_fermi(cmd, shader, NULL, root, root_size, group_count,
                             false, 0);
      return;
   }

   uint64_t qmd_addr;
   VkResult result = nvk_cmd_upload_qmd(cmd, shader, NULL,
                                        root, root_size,
                                        group_count,
                                        &qmd_addr, NULL);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   if (shader != NULL) {
      nvk_device_ensure_slm(dev, shader->info.slm_size,
                                 shader->info.crs_size);
   }

   struct nv_push *p = nvk_cmd_buffer_push(cmd, 8);

   /* Internal shaders don't want conditional rendering */
   P_IMMD(p, NVA0C0, SET_RENDER_ENABLE_OVERRIDE, MODE_ALWAYS_RENDER);

   P_MTHD(p, NVA0C0, SEND_PCAS_A);
   P_NVA0C0_SEND_PCAS_A(p, qmd_addr >> 8);

   if (nvk_cmd_buffer_compute_cls(cmd) <= TURING_COMPUTE_A) {
      P_IMMD(p, NVA0C0, SEND_SIGNALING_PCAS_B, {
            .invalidate = INVALIDATE_TRUE,
            .schedule = SCHEDULE_TRUE
      });
   } else {
      P_IMMD(p, NVC6C0, SEND_SIGNALING_PCAS2_B,
             PCAS_ACTION_INVALIDATE_COPY_SCHEDULE);
   }

   P_IMMD(p, NVA0C0, SET_RENDER_ENABLE_OVERRIDE, MODE_USE_RENDER_ENABLE);
}

void
nvk_cmd_dispatch_shader(struct nvk_cmd_buffer *cmd,
                        struct nvk_shader *shader,
                        const void *push_data, size_t push_size,
                        uint32_t groupCountX,
                        uint32_t groupCountY,
                        uint32_t groupCountZ)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct nvk_root_descriptor_table root = {
      .cs.group_count = {
         groupCountX,
         groupCountY,
         groupCountZ,
      },
   };
   assert(push_size <= sizeof(root.push));
   memcpy(root.push, push_data, push_size);

   if (NAK_CAN_PRINTF) {
      struct nvkmd_mem *bo = (struct nvkmd_mem *)dev->printf.bo;
      root.printf_buffer_addr = bo->va->addr;
   }

   nvk_cmd_dispatch_with_root(cmd, shader, &root, sizeof(root),
                              groupCountX, groupCountY, groupCountZ);
}

static void
mme_store_global(struct mme_builder *b,
                 struct mme_value64 addr,
                 struct mme_value v)
{
   mme_mthd(b, NV9097_SET_REPORT_SEMAPHORE_A);
   mme_emit_addr64(b, addr);
   mme_emit(b, v);
   mme_emit(b, mme_imm(0x10000000));
}

static void
mme_store_global_vec3_free_addr(struct mme_builder *b,
                                struct mme_value64 addr,
                                uint32_t offset,
                                struct mme_value x,
                                struct mme_value y,
                                struct mme_value z)
{
   if (offset > 0)
      mme_add64_to(b, addr, addr, mme_imm64(offset));

   mme_store_global(b, addr, x);
   mme_add64_to(b, addr, addr, mme_imm64(4));
   mme_store_global(b, addr, y);
   mme_add64_to(b, addr, addr, mme_imm64(4));
   mme_store_global(b, addr, z);
   mme_free_reg64(b, addr);
}

static void
mme_store_root_desc_group_count(struct mme_builder *b,
                                struct mme_value64 root_desc_addr,
                                struct mme_value group_count_x,
                                struct mme_value group_count_y,
                                struct mme_value group_count_z)
{
   uint32_t root_desc_size_offset =
      offsetof(struct nvk_root_descriptor_table, cs.group_count);
   mme_store_global_vec3_free_addr(b, root_desc_addr,
                                   root_desc_size_offset,
                                   group_count_x,
                                   group_count_y,
                                   group_count_z);
}

static void
mme_store_qmd_dispatch_size(struct mme_builder *b,
                            struct mme_value64 qmd_addr,
                            struct mme_value group_count_x,
                            struct mme_value group_count_y,
                            struct mme_value group_count_z)
{
   struct nak_qmd_dispatch_size_layout qmd_size_layout =
      nak_get_qmd_dispatch_size_layout(b->devinfo);
   assert(qmd_size_layout.y_start == qmd_size_layout.x_start + 32);

   if (qmd_size_layout.z_start == qmd_size_layout.y_start + 32) {
      mme_store_global_vec3_free_addr(b, qmd_addr,
                                      qmd_size_layout.x_start / 8,
                                      group_count_x,
                                      group_count_y,
                                      group_count_z);
   } else {
      mme_add64_to(b, qmd_addr, qmd_addr,
                   mme_imm64(qmd_size_layout.x_start / 8));
      mme_store_global(b, qmd_addr, group_count_x);

      assert(qmd_size_layout.z_start == qmd_size_layout.y_start + 16);
      struct mme_value group_count_yz =
         mme_merge(b, group_count_y, group_count_z, 16, 16, 0);
      mme_add64_to(b, qmd_addr, qmd_addr, mme_imm64(4));
      mme_store_global(b, qmd_addr, group_count_yz);
      mme_free_reg(b, group_count_yz);

      mme_free_reg64(b, qmd_addr);
   };
}

void
nvk_mme_dispatch_indirect(struct mme_builder *b)
{
   /* Fermi's compute class launches through methods, not through a launch
    * descriptor: here the group counts only go to the root table and to
    * the invocation counter; nvk_mme_fermi_launch_grid launches.
    */
   if (b->devinfo->cls_compute < KEPLER_COMPUTE_A) {
      struct mme_value group_count_x = mme_load(b);
      struct mme_value group_count_y = mme_load(b);
      struct mme_value group_count_z = mme_load(b);

      struct mme_value64 root_desc_addr = mme_load_addr64(b);
      mme_store_root_desc_group_count(b, root_desc_addr,
                                      group_count_x,
                                      group_count_y,
                                      group_count_z);

      /* Y and Z are 16b, so this cant't overflow */
      struct mme_value cs1 =
         mme_mul_32x32_32_free_srcs(b, group_count_y, group_count_z);
      struct mme_value64 cs2 =
         mme_umul_32x32_64_free_srcs(b, group_count_x, cs1);
      struct mme_value local_size = mme_load(b);
      struct mme_value64 count =
         mme_umul_32x64_64_free_srcs(b, local_size, cs2);

      nvk_build_mme_add_cs_invocations(b, count);
      return;
   }

   if (b->devinfo->cls_eng3d >= TURING_A) {
      /* Load everything before we switch to an indirect read */
      struct mme_value64 dispatch_addr = mme_load_addr64(b);
      struct mme_value64 root_desc_addr = mme_load_addr64(b);
      struct mme_value64 qmd_addr = mme_load_addr64(b);
      struct mme_value local_size = mme_load(b);

      mme_tu104_read_fifoed(b, dispatch_addr, mme_imm(3));
      mme_free_reg64(b, dispatch_addr);
      struct mme_value group_count_x = mme_load(b);
      struct mme_value group_count_y = mme_load(b);
      struct mme_value group_count_z = mme_load(b);

      mme_store_root_desc_group_count(b, root_desc_addr,
                                      group_count_x,
                                      group_count_y,
                                      group_count_z);

      mme_store_qmd_dispatch_size(b, qmd_addr,
                                  group_count_x,
                                  group_count_y,
                                  group_count_z);

      struct mme_value64 cs1 = mme_umul_32x32_64(b, group_count_y,
                                                    group_count_z);
      struct mme_value64 cs2 = mme_umul_32x32_64(b, group_count_x,
                                                    local_size);
      struct mme_value64 count = mme_mul64(b, cs1, cs2);
      mme_free_reg64(b, cs1);
      mme_free_reg64(b, cs2);

      nvk_build_mme_add_cs_invocations(b, count);
   } else {
      struct mme_value group_count_x = mme_load(b);
      struct mme_value group_count_y = mme_load(b);
      struct mme_value group_count_z = mme_load(b);

      struct mme_value64 root_desc_addr = mme_load_addr64(b);
      mme_store_root_desc_group_count(b, root_desc_addr,
                                      group_count_x,
                                      group_count_y,
                                      group_count_z);

      struct mme_value64 qmd_addr = mme_load_addr64(b);
      mme_store_qmd_dispatch_size(b, qmd_addr,
                                  group_count_x,
                                  group_count_y,
                                  group_count_z);

      /* Y and Z are 16b, so this cant't overflow */
      struct mme_value cs1 =
         mme_mul_32x32_32_free_srcs(b, group_count_y, group_count_z);
      struct mme_value64 cs2 =
         mme_umul_32x32_64_free_srcs(b, group_count_x, cs1);
      struct mme_value local_size = mme_load(b);
      struct mme_value64 count =
         mme_umul_32x64_64_free_srcs(b, local_size, cs2);

      nvk_build_mme_add_cs_invocations(b, count);
   }
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdDispatchIndirect2KHR(VkCommandBuffer commandBuffer,
                            const VkDispatchIndirect2InfoKHR* pInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);

   uint64_t dispatch_addr = pInfo->addressRange.address;

   if (nvk_cmd_buffer_compute_cls(cmd) < KEPLER_COMPUTE_A) {
      /* the counts reach the root table and the launch through macros */
      uint32_t zero[3] = { 0, 0, 0 };
      nvk_flush_compute_state(cmd, zero, zero);

      const struct nvk_descriptor_state *desc = &cmd->state.cs.descriptors;
      nvk_cmd_dispatch_fermi(cmd, cmd->state.cs.shader, desc,
                             desc->root, sizeof(desc->root), zero,
                             true, dispatch_addr);
      return;
   }

   /* We set these through the MME */
   uint32_t base_workgroup[3] = { 0, 0, 0 };
   uint32_t global_size[3] = { 0, 0, 0 };
   nvk_flush_compute_state(cmd, base_workgroup, global_size);

   uint64_t qmd_addr = 0, root_desc_addr = 0;
   VkResult result = nvk_cmd_flush_cs_qmd(cmd, &cmd->state, global_size,
                                          &qmd_addr, &root_desc_addr);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   struct nv_push *p;
   if (nvk_cmd_buffer_compute_cls(cmd) >= TURING_COMPUTE_A) {
      p = nvk_cmd_buffer_push(cmd, 14);
      if (nvk_cmd_buffer_compute_cls(cmd) < BLACKWELL_COMPUTE_A) {
         P_IMMD(p, NVC597, SET_MME_DATA_FIFO_CONFIG, FIFO_SIZE_SIZE_4KB);
      } else {
         /* The line in the other side of the if causes an implicit wfi
          * due to the subc switch which we apparently rely on for
          * correctness (!?). Anyway, to prevent issues on blackwell
          * we need to wfi here too.
          * TODO: delete this
          */
         P_IMMD(p, NVC86F, WFI, 0);
      }

      if (nvk_cmd_buffer_compute_cls(cmd) >= AMPERE_COMPUTE_B)
         P_1INC(p, NVC7C0, CALL_MME_MACRO(NVK_MME_DISPATCH_INDIRECT));
      else
         P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_DISPATCH_INDIRECT));
      P_INLINE_DATA(p, dispatch_addr >> 32);
      P_INLINE_DATA(p, dispatch_addr);
      P_INLINE_DATA(p, root_desc_addr >> 32);
      P_INLINE_DATA(p, root_desc_addr);
      P_INLINE_DATA(p, qmd_addr >> 32);
      P_INLINE_DATA(p, qmd_addr);
      P_INLINE_DATA(p, nvk_compute_local_size(cmd));
   } else {
      p = nvk_cmd_buffer_push(cmd, 5);
      /* Stall the command streamer */
      if (pdev->info.cls_compute >= HOPPER_COMPUTE_A) {
         P_IMMD(p, NVC86F, WFI, 0);
      } else {
         __push_immd(p, SUBC_NV9097, NV906F_SET_REFERENCE, 0);
      }

      P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_DISPATCH_INDIRECT));
      nv_push_update_count(p, sizeof(VkDispatchIndirectCommand) / 4);
      nvk_cmd_buffer_push_indirect(cmd, dispatch_addr, sizeof(VkDispatchIndirectCommand));
      p = nvk_cmd_buffer_push(cmd, 9);
      P_INLINE_DATA(p, root_desc_addr >> 32);
      P_INLINE_DATA(p, root_desc_addr);
      P_INLINE_DATA(p, qmd_addr >> 32);
      P_INLINE_DATA(p, qmd_addr);
      P_INLINE_DATA(p, nvk_compute_local_size(cmd));
   }

   P_MTHD(p, NVA0C0, SEND_PCAS_A);
   P_NVA0C0_SEND_PCAS_A(p, qmd_addr >> 8);
   if (nvk_cmd_buffer_compute_cls(cmd) <= TURING_COMPUTE_A) {
      P_IMMD(p, NVA0C0, SEND_SIGNALING_PCAS_B, {
            .invalidate = INVALIDATE_TRUE,
            .schedule = SCHEDULE_TRUE
      });
   } else {
      P_IMMD(p, NVC6C0, SEND_SIGNALING_PCAS2_B,
             PCAS_ACTION_INVALIDATE_COPY_SCHEDULE);
   }
}
