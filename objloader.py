import pyglet
from pyglet.gl import *
from pyglet import shapes
import numpy as np
import ctypes
import pywavefront

# Vertex shader source code
vertex_shader_code = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# Fragment shader source code
fragment_shader_code = """
#version 330 core
out vec4 fragColor;
void main()
{
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    
    # Convert the source code into a C-compatible string
    source = ctypes.c_char_p(source.encode('utf-8'))
    
    # Specify the number of strings, the array of strings, and the lengths of each string
    glShaderSource(shader, 1, ctypes.byref(source), None)
    
    glCompileShader(shader)
    
    # Check for compilation errors
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error_message = glGetShaderInfoLog(shader).decode('utf-8')
        raise RuntimeError(f"Shader compilation failed: {error_message}")
    
    return shader

def create_shader_program():
    vertex_shader = compile_shader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_code, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Check for linking errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error_message = glGetProgramInfoLog(program).decode('utf-8')
        raise RuntimeError(f"Shader program linking failed: {error_message}")

    return program

class OBJModel:
    def __init__(self, filename):
        # Load the OBJ file
        self.scene = pywavefront.Wavefront(filename, create_materials=True, collect_faces=True)
        
        # Extract vertices and indices
        self.vertices = []
        self.indices = []

        for name, mesh in self.scene.meshes.items():
            for material in mesh.materials:
                start = len(self.vertices) // 3
                self.vertices.extend(material.vertices)
                self.indices.extend([index + start for index in material.faces])

        # Convert to numpy arrays
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)

        # Create VAO and VBO
        self.vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)

        self.vbo = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.ebo = GLuint()
        glGenBuffers(1, ctypes.byref(self.ebo))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        position = glGetAttribLocation(shader_program, 'position')
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(position)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

def create_matrix_perspective(fovy, aspect, znear, zfar):
    h = np.tan(np.radians(fovy) / 2) * znear
    w = h * aspect
    return np.array([
        [znear / w, 0, 0, 0],
        [0, znear / h, 0, 0],
        [0, 0, -(zfar + znear) / (zfar - znear), -1],
        [0, 0, -2 * zfar * znear / (zfar - znear), 0]
    ], dtype=np.float32)

def create_matrix_view(position, target, up):
    zaxis = (position - target) / np.linalg.norm(position - target)
    xaxis = np.cross(up, zaxis) / np.linalg.norm(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)
    return np.array([
        [xaxis[0], yaxis[0], zaxis[0], 0],
        [xaxis[1], yaxis[1], zaxis[1], 0],
        [xaxis[2], yaxis[2], zaxis[2], 0],
        [-np.dot(xaxis, position), -np.dot(yaxis, position), -np.dot(zaxis, position), 1]
    ], dtype=np.float32)

def create_matrix_model(rotation):
    return np.array([
        [np.cos(rotation), 0, np.sin(rotation), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation), 0, np.cos(rotation), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

# Initialize the window
window = pyglet.window.Window(width=800, height=600, caption='OBJ Loader')

# Compile and link shaders
shader_program = create_shader_program()

# Load the OBJ model
obj_model = OBJModel('model.obj')

# Configure the OpenGL settings
glEnable(GL_DEPTH_TEST)

# Create projection and view matrices
projection = create_matrix_perspective(45.0, window.width / window.height, 0.1, 100.0)
view = create_matrix_view(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

# Get uniform locations
glUseProgram(shader_program)
projection_loc = glGetUniformLocation(shader_program, 'projection')
view_loc = glGetUniformLocation(shader_program, 'view')
model_loc = glGetUniformLocation(shader_program, 'model')

# Pass projection and view matrices to the shader
glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

# Create a rotation variable
rotation = 0.0

@window.event
def on_draw():
    global rotation

    # Clear the window
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Use the shader program
    glUseProgram(shader_program)
    
    # Create and pass the model matrix to the shader
    model = create_matrix_model(rotation)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    
    # Draw the OBJ model
    obj_model.draw()
    
    # Increment the rotation for the next frame
    rotation += 0.01

# Start the pyglet application
pyglet.app.run()
