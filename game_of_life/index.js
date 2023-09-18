var canvas = document.querySelector("canvas");
if (!canvas) throw new Error("Canvas with id water not found");

if (!navigator.gpu) throw new Error("WebGPU not supported");

const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance"
});
if (!adapter) throw new Error("Adapter not found");

const device = await adapter.requestDevice();

const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat
});

// data
const GRID_SIZE = 1000;
const WORKGROUP_SIZE = 16;
const QUAD_VERTICES = new Float32Array([
    -1, -1,
     1, -1,
     1,  1,

    -1, -1,
     1,  1,
    -1,  1
]).map(x => x * 0.8);
const UPDATE_INTERVAL = 1;
let step = 0; // dynamic

// build vertex buffer
const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: QUAD_VERTICES.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
const vertexBufferLayout = {
    arrayStride: 2 * 4,
    attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0,
    }]
};
device.queue.writeBuffer(vertexBuffer, 0, QUAD_VERTICES);

// build uniform buffer 
const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

// cell state buffer
const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
const cellStateStorage = [
    device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }), 
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
];

// init cell state buffers
for (let i = 0; i < cellStateArray.length; i++)
    cellStateArray[i] = Math.random() > 0.4 ? 1 : 0;
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

// layouts
const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: {}
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
    }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
    }]
});

const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [ bindGroupLayout ]
});

const cellShaderModule = device.createShaderModule({
    label: 'Cell shader',
    code : `
    struct VertexInput {
        @location(0) pos: vec2f,
        @builtin(instance_index) instance: u32,
    };

    struct VertexOutput {
        @builtin(position) pos: vec4f,
        @location(0) cell: vec2f,
    }

    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage> cellState: array<u32>;

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
        let i = f32(input.instance);
        let cell = vec2f(i % grid.x, floor(i / grid.y));
        let scale = f32(cellState[input.instance]);

        let cellOffset = cell / grid * 2;
        let gridPos = (scale * input.pos + 1) / grid - 1 + cellOffset;

        var output: VertexOutput;
        output.pos = vec4f(gridPos, 0.0, 1.0);
        output.cell = cell / grid;
        return output;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
        let c = input.cell;
        return vec4f(c, 1 - c.x, 1);
    }
    `
});

const simulationShaderModule = device.createShaderModule({
    label: "Game of Life simulation shader",
    code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;
        
        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        fn cellIndex(cell: vec2u) -> u32 {
            return (cell.x % u32(grid.x)) + 
                   (cell.y % u32(grid.y)) * u32(grid.x);
        }

        fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2u(x, y))];
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            let activeNeighbors = cellActive(cell.x + 1, cell.y + 1) 
                                + cellActive(cell.x + 1, cell.y)
                                + cellActive(cell.x + 1, cell.y - 1)
                                + cellActive(cell.x, cell.y - 1)
                                + cellActive(cell.x, cell.y + 1)
                                + cellActive(cell.x - 1, cell.y - 1)
                                + cellActive(cell.x - 1, cell.y)
                                + cellActive(cell.x - 1, cell.y + 1);

            let i = cellIndex(cell.xy);

            switch activeNeighbors {
                case 2: {
                    cellStateOut[i] = cellStateIn[i];
                }
                case 3: {
                    cellStateOut[i] = 1;
                }
                default: {
                    cellStateOut[i] = 0;
                }
            }
        }
    `
});

const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout],
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }
});

const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    }
});

const bindGroups = [
    device.createBindGroup({
      label: "Cell renderer bind group A",
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: uniformBuffer }
      }, {
        binding: 1,
        resource: { buffer: cellStateStorage[0] }
      }, {
        binding: 2,
        resource: { buffer: cellStateStorage[1] }
      }],
    }),
    device.createBindGroup({
      label: "Cell renderer bind group B",
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: uniformBuffer }
      }, {
        binding: 1,
        resource: { buffer: cellStateStorage[1] }
      }, {
        binding: 2,
        resource: { buffer: cellStateStorage[0] }
      }],
    })
];

function updateGrid() {
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
    computePass.end();

    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0.0, g: 0.0, b: 0.4, a: 1.0},
            storeOp: "store",
        }]
    }); 
    pass.setPipeline(cellPipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.draw(QUAD_VERTICES.length / 2, GRID_SIZE * GRID_SIZE);
    pass.end();

    step++;

    device.queue.submit([encoder.finish()]);
}

setInterval(updateGrid, UPDATE_INTERVAL);