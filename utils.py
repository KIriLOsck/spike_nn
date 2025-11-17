import math, random
import OpenGL.GL as gl

def get_circle_point(x, y, i, t):
    r = 0.1 + (i % 30) / 100
    t *= (i % 10) if (i % 10) else 1
    d = 2 * math.pi * math.cos(t + i) + 0.1

    new_x, new_y = r * math.cos(d), r * math.sin(d)

    if i % 4 == 1:
        new_x += 0.2
    elif i % 4 == 2:
        new_x -= 0.2
    elif i % 4 == 3:
        new_y += 0.2
    else:
        new_y -= 0.2
    
    return (new_x, new_y)

def generate_circles(n, d):
    points = []
    max_attempts = 1000  # Максимальное количество попыток для каждой точки
    
    for _ in range(n):
        attempts = 0
        while attempts < max_attempts:
            # Генерируем случайные координаты в нормализованном диапазоне
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            
            # Проверяем, что точка внутри единичного круга
            if math.sqrt(x**2 + y**2) <= 0.5:
                # Проверяем минимальное расстояние до других точек
                valid = True
                for point in points:
                    dx = point[0] - x
                    dy = point[1] - y
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < d:
                        valid = False
                        break
                
                if valid:
                    points.append((x, y))
                    break
            
            attempts += 1
    
    return points

def compile_shader(source, shader_type):
    """Компилирует шейдер и проверяет ошибки"""
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(shader).decode()
        raise Exception(f"Ошибка компиляции шейдера: {error}")
    
    return shader