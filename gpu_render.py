import math
import time
import OpenGL.GL as gl
from OpenGL.GL import glGetUniformLocation, glUniform1f
import glfw
import numpy as np
from ctypes import c_void_p, sizeof

from nn import SpikeNeuralNetwork
from utils import generate_circles, get_circle_point, compile_shader


WHITE = (1, 1, 1)
BLUE = (1, 0, 0)
RED = (0, 1, 0)
BLACK = (0, 0, 0)

NEURON = (90/255, 90/255, 90/255, 3.0)
NEURON_L = (1.0, 200/255, 55/255, 1.0)

HEIGHT, WIDTH = 1000, 1900

NEURONS_COUNT = 1000

def update_points(time_q, nn, points=None):

        if points is None:
            coords = generate_circles(NEURONS_COUNT, 0.0001)
            points = [(x, y, *NEURON_L, i) if i in nn.active_neurons else (x, y, *NEURON, i) for i, (x, y) in enumerate(coords)]
            new_points = points
        else:
            new_points = []
            for i, (x, y, r, g, b, s) in enumerate(points):
                if i in nn.active_neurons:
                    new_points.append((*get_circle_point(x, y, i, time_q)[:2], *NEURON_L))
                else:
                    new_points.append((*get_circle_point(x, y, i, time_q)[:2], *NEURON))
        return np.array(new_points, dtype=np.float32)

def main():

    points = None
    time_val = 0.1

    # Инициализация GLFW =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if not glfw.init():
        exit()

    window = glfw.create_window(WIDTH, HEIGHT, "Neural Network Visualization", None, None)
    nn = SpikeNeuralNetwork(NEURONS_COUNT)

    if not window:
        glfw.terminate()
        exit()

    glfw.make_context_current(window)

    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
    gl.glEnable(gl.GL_POINT_SMOOTH)

    # r = 0.1 + (i % 30) / 100
    # t *= (i % 10) if (i % 10) else 1
    # d = 2 * math.pi * math.cos(t + i) + 0.1

    # new_x, new_y = r * math.cos(d), r * math.sin(d)

    # if i % 4 == 1:
    #     new_x += 0.2
    # elif i % 4 == 2:
    #     new_x -= 0.2
    # elif i % 4 == 3:
    #     new_y += 0.2
    # else:
    #     new_y -= 0.2

    vertex_shader_source = vertex_shader_source = """
    #version 330 core
    layout (location = 0) in vec2 aPos;  // Можно оставить для совместимости, но не используем
    layout (location = 1) in vec3 aColor;
    layout (location = 2) in vec2 aSizeIndex; // x: size, y: index

    out vec3 ourColor;
    uniform float timeValue;

    const float PI = 3.141592653589793;

    void main() {
        float i = aSizeIndex.y; // индекс точки
        float r = 0.1 + mod(i, 30.0) / 100.0;
        
        // Вычисляем модифицированное время для этой точки
        float modifiedTime = timeValue;
        if (mod(i, 10.0) != 0.0) {
            modifiedTime *= mod(i, 10.0);
        }
        // else - оставляем как есть (умножение на 1)
        
        // Основное вычисление позиции
        float d = 2.0 * PI * cos(modifiedTime + i) + 0.1;
        vec2 circle_pos = vec2(r * cos(d), r * sin(d));
        
        // Смещения в зависимости от индекса
        vec2 offset = vec2(0.0);
        float index_mod = mod(i, 4.0);
        if (index_mod == 1.0) {
            offset = vec2(0.2, 0.0);
        } else if (index_mod == 2.0) {
            offset = vec2(-0.2, 0.0);
        } else if (index_mod == 3.0) {
            offset = vec2(0.0, 0.2);
        } else {
            offset = vec2(0.0, -0.2);
        }
        
        vec2 final_pos = circle_pos + offset + aPos;
        gl_Position = vec4(final_pos, 0.0, 1.0);
        gl_PointSize = aSizeIndex.x;
        ourColor = aColor;
    }
"""

    fragment_shader_source = """
    #version 330 core
    out vec4 FragColor;
    in vec3 ourColor;
    void main() {
        FragColor = vec4(ourColor, 1.0);
    }
    """

    vertex_shader = compile_shader(vertex_shader_source, gl.GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)

    shader_program = gl.glCreateProgram()

    gl.glAttachShader(shader_program, vertex_shader)
    gl.glAttachShader(shader_program, fragment_shader)
    gl.glLinkProgram(shader_program)

    VAO = gl.glGenVertexArrays(1)
    VBO = gl.glGenBuffers(1)

    gl.glBindVertexArray(VAO)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)

    initial_points = np.array([(0,0,0,0,0,0,0)] * len(nn), dtype=np.float32)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, initial_points.nbytes, initial_points, gl.GL_DYNAMIC_DRAW)

    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 7 * sizeof(gl.GLfloat), c_void_p(0))
    gl.glEnableVertexAttribArray(0)

    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 7 * sizeof(gl.GLfloat), c_void_p(2 * sizeof(gl.GLfloat)))
    gl.glEnableVertexAttribArray(1)

    gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 7 * sizeof(gl.GLfloat), c_void_p(5 * sizeof(gl.GLfloat)))
    gl.glEnableVertexAttribArray(2)

    gl.glUseProgram(shader_program)
    time_value_loc = glGetUniformLocation(shader_program, "timeValue")

    points = update_points(time_val, nn, points)
    print(points[:4])

    while not glfw.window_should_close(window):
        # Обработка событий
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # nn.neurons[1].value += 0.01
        # nn.neurons[1].reLU()

        glUniform1f(time_value_loc, time_val)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, points.nbytes, points)

        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glBindVertexArray(VAO)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(points))

        # Рисование связей между нейронами
        # for i, neuron in enumerate(nn):
        #     for link in neuron.links:
        #         target_index, _, weight = link
        #         col = abs(weight)
        #         if weight > 0:
        #             color = (240/255, col, col)
        #         elif weight < 0:
        #             color = (col, col, 240/255)
        #         else:
        #             color = BLACK

        #         line_width = 2 if abs(weight) > 0.5 else 1
                # draw_line(circles[i], circles[target_index], color, line_width)

        time_val += 0.000001

        glfw.swap_buffers(window)
        glfw.poll_events()

    gl.glDeleteVertexArrays(1, [VAO])
    gl.glDeleteBuffers(1, [VBO])
    gl.glDeleteProgram(shader_program)
    glfw.terminate()


if __name__ == "__main__":
    main()