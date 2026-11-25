# wignix-mesa

Mesa with the WigNix nouveau work: occlusion queries and a threaded
context for the nvc0 gallium driver, and a Vulkan (NVK) port to Fermi.

Based on Mesa 26.2.7 (`mesa-26.2.7`). The commits on top are:

- **nvc0: ZCULL for GF106 (Quadro 2000M)** — occlusion queries for the
  Fermi GF106 with multiple regions, MSAA and layered depth, and
  save/restore of ZCULL state.
- **nvc0: submit when needed, not after every draw** — submit work on
  demand (`NVC0_KICK_DRAWS`, default 128), place streamed dynamic buffers
  in system memory, and fence index buffers on every draw.
- **nvc0: threaded context** — a `u_threaded_context` path, opt-in with
  `NVC0_THREADED=1`.
- **nvc0: ZCULL and threaded context on by default** — with
  `NVC0_ZCULL=0` and `NVC0_THREADED=0` to turn them off per process.
- **nouveau: Vulkan (NVK) on Fermi** — ports NVK to the Quadro 2000M at
  Vulkan 1.1, enough for WebGPU through Dawn in Chromium.

The Vulkan port needs the copy-engine fix in `linux-sandy` 7.3.1_16 or
later (see `wignix-kernel`). Notes and tests for the Vulkan work live in
`wignix-nouveau`.

## Why

The Quadro 2000M (GF106) is a Fermi card. No vendor ships Vulkan on
Fermi; this tree makes the open NVK driver run on it. On the same card
the nvc0 changes give occlusion queries and remove the per-draw submit
stall.

## Building

Apply these commits to a Mesa checkout at `mesa-26.2.7`, or use them as
the patch set of the `mesa` package in `wignix-packages`:

```
git format-patch mesa-26.2.7..sandy -- src
```
