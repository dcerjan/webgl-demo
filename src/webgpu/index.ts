import { useLayoutEffect, useRef, useState } from "react"

const vertexShaderSource = /* wgsl */`
struct VSOut {
  @builtin(position) position: vec4<f32>;
  @location(0) color: vec3<f32>;
}

struct UBO {
  modelViewProjection: mat4x4<f32>;
  primaryColor: vec4<f32>;
  accentColor: vec4<f32>;
  angle: f32;
};
@group(0)
@binding(0)
var<uniform> uniforms: UBO;

@stage(vertex)
fn main(
  @location(0) inPos: vec3<f32>,
  @location(1) inColor: vec3<f32>
) -> VSOut {
  var a = uniforms.angle;
  var vsOut: VSOut;
  var rotMatrix = mat4x4<f32>(
     cos(a), sin(a), 0.0, 0.0,
    -sin(a), cos(a), 0.0, 0.0,
        0.0,    0.0, 1.0, 0.0,
        0.0,    0.0, 0.0, 1.0
  );
  vsOut.position = uniforms.modelViewProjection * rotMatrix * vec4<f32>(inPos, 1.0);
  vsOut.color = inColor;

  return vsOut;
}
`

const fragShaderSource = /* wgsl */`
@stage(fragment)
fn main(
  @location(0) inColor: vec3<f32>
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, 1.0);
}
`


const init = async (canvas: HTMLCanvasElement): Promise<GPUCanvasContext> => {
  const entry: GPU = navigator.gpu
  if (entry == null) {
    throw new Error('WebGPU is not supported!')
  }

  const adapter = await entry.requestAdapter({ powerPreference: 'high-performance' })

  const device = await adapter!.requestDevice()

  const queue = device.queue

  const contextConfig: GPUCanvasConfiguration = {
    device,
    format: 'bgra8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
  }

  const context = canvas.getContext('webgpu')

  if (context == null) {
    throw new Error('Error creating \'webgpu\' context')
  }

  context.configure(contextConfig)

  const depthTextureCDescriptor: GPUTextureDescriptor = {
    size: [canvas.width, canvas.height, 1],
    dimension: '2d',
    format: 'depth24plus-stencil8',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
  }
  const depthTexture = device.createTexture(depthTextureCDescriptor)
  const depthTextureView = depthTexture.createView()

  const positions = new Float32Array([
    1.0, -1.0, 0.0,
    -1.0, -1.0, 0.0,
    0.0,  1.0, 0.0
  ])

  const colors = new Float32Array([
    1.0, 0.0, 0.0, // 🔴
    0.0, 1.0, 0.0, // 🟢
    0.0, 0.0, 1.0  // 🔵
  ])

  const indices = new Uint16Array([ 0, 1, 2 ])

  const createBuffer = (arr: Float32Array | Uint16Array, usage: number) => {
    //  Align to 4 bytes
    const desc: GPUBufferDescriptor = {
      size: ((arr.byteLength + 3) & ~3),
      usage,
      mappedAtCreation: true,
    }
    const buffer = device.createBuffer(desc)
    const writeArray = arr instanceof Uint16Array
      ? new Uint16Array(buffer.getMappedRange())
      : new Float32Array(buffer.getMappedRange())

    writeArray.set(arr)
    buffer.unmap()
    return buffer
  }

  const positionBuffer = createBuffer(positions, GPUBufferUsage.VERTEX)
  const colorBuffer = createBuffer(colors, GPUBufferUsage.VERTEX)
  const indexBuffer = createBuffer(indices, GPUBufferUsage.INDEX)

  const vertModule = device.createShaderModule({ code: vertexShaderSource })
  const fragModule = device.createShaderModule({ code: fragShaderSource })

  const uniformData = new Float32Array([
    // modelViewProjection matrix
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 2.0,

    // primary Color
    0.9, 0.1, 0.3, 1.0,

    // accent Color
    0.8, 0.2, 0.8, 1.0,

    // rotation
    0.0,
  ])

  const uniformBuffer = createBuffer(uniformData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)

  const uniformBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        type: 'uniform' as const
      }
    }],
  })

  const uniformBindGroup = device.createBindGroup({
    layout: uniformBindGroupLayout,
    entries: [{
      binding: 0,
      resource: {
        buffer: uniformBuffer,
      }
    }]
  })

  const pipelineLayoutDescriptor: GPUPipelineLayoutDescriptor = {
    bindGroupLayouts: [uniformBindGroupLayout]
  }
  const layout = device.createPipelineLayout(pipelineLayoutDescriptor)

  const positionAttribDesc: GPUVertexAttribute = {
    shaderLocation: 0, // [[location(0)]]
    offset: 0,
    format: 'float32x3',
  }

  const colorAttribDesc: GPUVertexAttribute = {
    shaderLocation: 1, // [[location(1)]]
    offset: 0,
    format: 'float32x3',
  }

  const positionBufferDesc: GPUVertexBufferLayout = {
    attributes: [positionAttribDesc],
    arrayStride: 4 * 3, // sizeof(float) * 3
    stepMode: 'vertex',
  }

  const colorBufferDesc: GPUVertexBufferLayout = {
    attributes: [colorAttribDesc],
    arrayStride: 4 * 3, // sizeof(float) * 3
    stepMode: 'vertex'
  }

  const depthStencil: GPUDepthStencilState = {
    depthWriteEnabled: true,
    depthCompare: 'less',
    format: 'depth24plus-stencil8',
  }

  const vertex: GPUVertexState = {
    module: vertModule,
    entryPoint: 'main',
    buffers: [positionBufferDesc, colorBufferDesc]
  }

  const colorState: GPUColorTargetState = {
    format: 'bgra8unorm',
  }

  const fragment: GPUFragmentState = {
    module: fragModule,
    entryPoint: 'main',
    targets: [colorState],
  }

  const primitive: GPUPrimitiveState = {
    frontFace: 'cw',
    cullMode: 'none',
    topology: 'triangle-list',
  }

  const pipeline = device.createRenderPipeline({
    layout,
    vertex,
    fragment,
    primitive,
    depthStencil,
  })

  const encodeCommands = (colorTextureView: GPUTextureView, angle: number) => {
    const colorAttachment: GPURenderPassColorAttachment = {
      view: colorTextureView,
      // loadValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    } as any

    const depthAttachment: GPURenderPassDepthStencilAttachment = {
      view: depthTextureView,
      // depthLoadValue: 1,
      depthLoadOp: 'clear',
      depthClearValue: 1,
      depthStoreOp: 'store',
      // stencilLoadValue: 'load',
      stencilLoadOp: 'load',
      stencilClearValue: 0,
      stencilStoreOp: 'store',
    } as any

    const renderPassDesc: GPURenderPassDescriptor = {
      colorAttachments: [colorAttachment],
      depthStencilAttachment: depthAttachment,
    }

    const commandEncoder = device.createCommandEncoder()

    // Update angle every frame
    const uploadBuffer = device.createBuffer({
      size: 1 * 4,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    })
    new Float32Array(uploadBuffer.getMappedRange()).set([angle])
    uploadBuffer.unmap()
    commandEncoder.copyBufferToBuffer(
      uploadBuffer,
      0, // src start offset
      uniformBuffer,
      24 * 4, // dest start offset (write into uniforms.angle)
      1 * 4, // number of bytes to copy over (sizeof f32)
    )

    const renderPass = commandEncoder.beginRenderPass(renderPassDesc)
    renderPass.setPipeline(pipeline)
    renderPass.setViewport(0, 0, canvas.width, canvas.height, 0, 1)
    renderPass.setScissorRect(0, 0, canvas.width, canvas.height)
    renderPass.setBindGroup(0, uniformBindGroup)
    renderPass.setVertexBuffer(0, positionBuffer)
    renderPass.setVertexBuffer(1, colorBuffer)
    renderPass.setIndexBuffer(indexBuffer, 'uint16')
    renderPass.drawIndexed(3)
    renderPass.endPass()

    queue.submit([commandEncoder.finish()])
  }

  const render = () => {
    const colorTexture = context.getCurrentTexture()
    const colorTextureView = colorTexture.createView()

    const angle = window.performance.now() * 0.001
    encodeCommands(colorTextureView, angle)

    requestAnimationFrame(render)
  }

  render()

  return context
}

export const useWebGpu = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [context, setContext] = useState<GPUCanvasContext | null>(null)

  useLayoutEffect(() => {
    // initialize once
    if (canvasRef.current != null && context == null) {
      init(canvasRef.current).then(setContext)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return canvasRef
}