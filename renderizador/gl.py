#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    supersample_scale = 2
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    lights = []   # lista de luzes na cena

    @staticmethod
    def setup(width, height, near=0.01, far=1000, supersample_scale=2):
        """Initializes GL state and dimensions."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.supersample_scale = supersample_scale
        GL.render_width = width * supersample_scale
        GL.render_height = height * supersample_scale
        GL.ambient_light = [0.2, 0.2, 0.2]  # Default ambient light

        # Reset matrices for a new render
        GL.matrix_stack = []
        GL.model_matrix = np.identity(4)
        GL.view_matrix = np.identity(4)
        GL.projection_matrix = np.identity(4)


    @staticmethod
    def _transform_vertex(vertex):
        """Transforms a 3D vertex to 2D screen coordinates and preserves Z and W."""
        vec = np.array([vertex[0], vertex[1], vertex[2], 1.0])

        # Apply model, view, and projection matrices
        if hasattr(GL, "model_matrix"):
            vec = GL.model_matrix @ vec
        if hasattr(GL, "view_matrix"):
            vec = GL.view_matrix @ vec
        if hasattr(GL, "projection_matrix"):
            vec = GL.projection_matrix @ vec

        # Store w from clip space, it's needed for perspective correction
        w_clip = vec[3]

        # Perspective divide to get Normalized Device Coordinates (NDC)
        if w_clip != 0:
            vec_ndc = vec / w_clip
        else:
            vec_ndc = vec

        # NDC → screen coordinates
        x = int((vec_ndc[0] * 0.5 + 0.5) * (GL.render_width - 1))
        y = int((vec_ndc[1] * 0.5 + 0.5) * (GL.render_height - 1))
        
        # The Z value in NDC is used for the depth buffer
        z_ndc = vec_ndc[2]

        return (x, y, z_ndc, w_clip)
    
    @staticmethod
    def _calculate_triangle_normal(v0, v1, v2):
        """Calculates the normal vector of a triangle using the cross product."""
        # Create two edge vectors from the vertices
        edge1 = np.array(v1) - np.array(v0)
        edge2 = np.array(v2) - np.array(v0)
        
        # The cross product gives a perpendicular vector (the normal)
        normal = np.cross(edge1, edge2)
        
        # Normalize the vector to have a length of 1
        norm = np.linalg.norm(normal)
        return (normal / norm).tolist() if norm > 0 else [0, 0, 1]
    
    @staticmethod
    def _calculate_lighting(normal, world_pos, material_colors):
        """
        Calculates the final color of a point using the Phong reflection model.
        """
        diffuse_color = material_colors.get("diffuseColor", [0.8, 0.8, 0.8])
        specular_color = material_colors.get("specularColor", [0.0, 0.0, 0.0])
        emissive_color = material_colors.get("emissiveColor", [0.0, 0.0, 0.0])
        shininess = material_colors.get("shininess", 0.2)
        ambient_intensity = material_colors.get("ambientIntensity", 0.2)

        if not GL.lights:
            ambient_contribution = np.array(GL.ambient_light) * np.array(diffuse_color) * ambient_intensity
            final_color = np.array(emissive_color) + ambient_contribution
            return np.clip(final_color, 0.0, 1.0).tolist()
        
        ambient_contribution = np.array(GL.ambient_light) * np.array(diffuse_color) * ambient_intensity
        final_color = np.array(emissive_color) + ambient_contribution

        normal = np.array(normal)
        norm_length = np.linalg.norm(normal)
        if norm_length > 0: normal /= norm_length

        # The view vector is from the point on the surface to the camera
        camera_pos = np.array(GL.eye)
        view_dir = camera_pos - world_pos
        view_dir /= np.linalg.norm(view_dir)

        for light in GL.lights:
            light_color = np.array(light['color'])
            light_dir = -np.array(light['direction'])
            light_dir /= np.linalg.norm(light_dir)

            diffuse_factor = max(0.0, np.dot(normal, light_dir))
            diffuse_term = light['intensity'] * light_color * np.array(diffuse_color) * diffuse_factor
            
            halfway_dir = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
            specular_factor = pow(max(0.0, np.dot(normal, halfway_dir)), shininess * 128)
            specular_term = light['intensity'] * light_color * np.array(specular_color) * specular_factor
            
            final_color += (diffuse_term + specular_term)

        return np.clip(final_color, 0.0, 1.0).tolist()


    @staticmethod
    def _draw_inside_triangle(p0, p1, p2, emissive_color):
        """Draws a filled triangle using scanline rasterization with edge functions."""
        v0_x, v0_y = p0
        v1_x, v1_y = p1
        v2_x, v2_y = p2

        # Bounding box
        min_x, max_x = min(v0_x, v1_x, v2_x), max(v0_x, v1_x, v2_x)
        min_y, max_y = min(v0_y, v1_y, v2_y), max(v0_y, v1_y, v2_y)

        def edge_func(ax, ay, bx, by, px, py):
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                w0 = edge_func(v1_x, v1_y, v2_x, v2_y, x, y)
                w1 = edge_func(v2_x, v2_y, v0_x, v0_y, x, y)
                w2 = edge_func(v0_x, v0_y, v1_x, v1_y, x, y)

                # Check if pixel is inside the triangle
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    if 0 <= x < GL.render_width and 0 <= y < GL.render_height:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8,
                                [emissive_color[0] * 255,
                                emissive_color[1] * 255,
                                emissive_color[2] * 255])


    @staticmethod
    def _draw_inside_triangle_color_and_tex(p0, p1, p2, c0, c1, c2, uv0, uv1, uv2, texture_img, default_color_tuple, transparency):
        """Draws a triangle with Z-buffering, alpha blending, and perspective-correct interpolation."""
        (x0, y0, z0, w_clip0) = p0
        (x1, y1, z1, w_clip1) = p1
        (x2, y2, z2, w_clip2) = p2

        min_x = max(0, int(min(x0, x1, x2)))
        max_x = min(GL.render_width - 1, int(max(x0, x1, x2)))
        min_y = max(0, int(min(y0, y1, y2)))
        max_y = min(GL.render_height - 1, int(max(y0, y1, y2)))

        def edge_func(ax, ay, bx, by, px, py):
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

        area = edge_func(x0, y0, x1, y1, x2, y2)
        if area == 0:
            return

        one_over_w0 = 1.0 / w_clip0 if w_clip0 != 0 else 0
        one_over_w1 = 1.0 / w_clip1 if w_clip1 != 0 else 0
        one_over_w2 = 1.0 / w_clip2 if w_clip2 != 0 else 0

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                w0_bary = edge_func(x1, y1, x2, y2, x, y)
                w1_bary = edge_func(x2, y2, x0, y0, x, y)
                w2_bary = edge_func(x0, y0, x1, y1, x, y)

                is_inside = (w0_bary >= 0 and w1_bary >= 0 and w2_bary >= 0) or \
                            (w0_bary <= 0 and w1_bary <= 0 and w2_bary <= 0)

                if is_inside:
                    b0, b1, b2 = w0_bary / area, w1_bary / area, w2_bary / area

                    interp_one_over_w = b0 * one_over_w0 + b1 * one_over_w1 + b2 * one_over_w2
                    if abs(interp_one_over_w) < 1e-9:
                        continue

                    z_over_w = b0 * z0 * one_over_w0 + b1 * z1 * one_over_w1 + b2 * z2 * one_over_w2
                    z_interp = z_over_w / interp_one_over_w
                    current_depth = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]
                    
                    if z_interp < current_depth:
                        # The depth test passed. Update the Z-buffer IMMEDIATELY for ALL objects.
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [z_interp])

                        # Now, calculate the pixel's color
                        source_color = default_color_tuple
                        if texture_img is not None and all(uv is not None for uv in [uv0, uv1, uv2]):
                            u_over_w = b0*uv0[0]*one_over_w0 + b1*uv1[0]*one_over_w1 + b2*uv2[0]*one_over_w2
                            v_over_w = b0*uv0[1]*one_over_w0 + b1*uv1[1]*one_over_w1 + b2*uv2[1]*one_over_w2
                            u, v = u_over_w / interp_one_over_w, v_over_w / interp_one_over_w
                            h, w, _ = texture_img.shape
                            tex_x, tex_y = int(v * (w - 1)), int((1 - u) * (h - 1))
                            px = texture_img[max(0, min(tex_y, h - 1)), max(0, min(tex_x, w - 1))]
                            source_color = [int(px[0]), int(px[1]), int(px[2])]
                        elif all(c is not None for c in [c0, c1, c2]):
                            r_over_w = b0*c0[0]*one_over_w0 + b1*c1[0]*one_over_w1 + b2*c2[0]*one_over_w2
                            g_over_w = b0*c0[1]*one_over_w0 + b1*c1[1]*one_over_w1 + b2*c2[1]*one_over_w2
                            b_over_w = b0*c0[2]*one_over_w0 + b1*c1[2]*one_over_w1 + b2*c2[2]*one_over_w2
                            r, g, b = r_over_w / interp_one_over_w, g_over_w / interp_one_over_w, b_over_w / interp_one_over_w
                            source_color = [int(r * 255), int(g * 255), int(b * 255)]

                        # Now, determine if blending is needed for the color
                        alpha = 1.0 - transparency
                        if alpha < 1.0: # Is the object transparent?
                            dest_color = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                            final_r = source_color[0] * alpha + dest_color[0] * (1.0 - alpha)
                            final_g = source_color[1] * alpha + dest_color[1] * (1.0 - alpha)
                            final_b = source_color[2] * alpha + dest_color[2] * (1.0 - alpha)
                            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [final_r, final_g, final_b])
                        else: # The object is opaque
                            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, source_color)


    @staticmethod
    def _draw_line(p0, p1, emissive_color):
        """Draws a line between two points using Bresenham's algorithm."""
        x0, y0 = p0
        x1, y1 = p1

        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < GL.render_width and 0 <= y0 < GL.render_height:
                gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, [emissive_color[0] * 255, emissive_color[1] * 255, emissive_color[2] * 255])
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    @staticmethod
    def _draw_line_3d(p0_data, p1_data, emissive_color):
        """Draws a line between two 3D-transformed points with Z-buffering."""
        (x0, y0, z0, w0) = p0_data
        (x1, y1, z1, w1) = p1_data

        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        # Determine the total number of steps for interpolation
        steps = max(dx, dy)
        if steps == 0:
            return

        # Bresenham's algorithm main loop
        while True:
            # Check if the current pixel is within the framebuffer bounds
            if 0 <= x0 < GL.render_width and 0 <= y0 < GL.render_height:
                # Interpolate Z value along the line
                # Heuristic: use distance from start point as a simple interpolation factor
                dist_from_start = max(abs(x0 - p0_data[0]), abs(y0 - p0_data[1]))
                t = dist_from_start / steps if steps > 0 else 0.0
                z_interp = z0 * (1.0 - t) + z1 * t

                # Depth Test
                current_depth = gpu.GPU.read_pixel([x0, y0], gpu.GPU.DEPTH_COMPONENT32F)[0]
                if z_interp < current_depth:
                    # If test passes, update depth and color buffers
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.DEPTH_COMPONENT32F, [z_interp])
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, [emissive_color[0] * 255, emissive_color[1] * 255, emissive_color[2] * 255])

            # Break condition for the loop
            if x0 == x1 and y0 == y1:
                break

            # Bresenham's algorithm step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    @staticmethod
    def _choose_mip_level(p0, p1, p2, uv0, uv1, uv2, base_texture, num_levels):
        """Calculates the appropriate Mipmap Level of Detail (LOD)."""
        (x0,y0,_,_), (x1,y1,_,_), (x2,y2,_,_) = p0, p1, p2
        
        du1, dv1 = uv1[0]-uv0[0], uv1[1]-uv0[1]
        du2, dv2 = uv2[0]-uv0[0], uv2[1]-uv0[1]
        dx1, dy1 = x1-x0, y1-y0
        dx2, dy2 = x2-x0, y2-y0
        
        den = dx1*dy2 - dx2*dy1
        if abs(den) < 1e-6: return 0
        
        dudx = (du1*dy2 - du2*dy1)/den
        dvdx = (dv1*dy2 - dv2*dy1)/den
        h, w, _ = base_texture.shape
        rho = math.sqrt(dudx**2 + dvdx**2) * w
        lod = math.log2(rho) if rho > 0 else 0
        level = int(round(lod))
        return max(0, min(level, num_levels - 1))
    
    @staticmethod
    def _draw_triangle_simple(p0, p1, p2, color_tuple, transparency):
            (x0,y0,z0,_), (x1,y1,z1,_), (x2,y2,z2,_) = p0, p1, p2
            
            min_x = max(0, int(min(x0, x1, x2)))
            max_x = min(GL.render_width - 1, int(max(x0, x1, x2)))
            min_y = max(0, int(min(y0, y1, y2)))
            max_y = min(GL.render_height - 1, int(max(y0, y1, y2)))

            def edge(ax, ay, bx, by, px, py): return (px-ax)*(by-ay)-(py-ay)*(bx-ax)
            area = edge(x0,y0,x1,y1,x2,y2)
            if abs(area) < 1e-6: return

            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    wA, wB, wC = edge(x1,y1,x2,y2,x,y), edge(x2,y2,x0,y0,x,y), edge(x0,y0,x1,y1,x,y)
                    is_inside = (wA >= 0 and wB >= 0 and wC >= 0) or (wA <= 0 and wB <= 0 and wC <= 0)
                    
                    if is_inside:
                        alpha, beta, gamma = wA/area, wB/area, wC/area
                        z_interp_ndc = alpha * z0 + beta * z1 + gamma * z2
                        
                        depth_val = (z_interp_ndc + 1.0) * 0.5
                        current_depth = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]

                        if depth_val < current_depth:
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.DEPTH_COMPONENT32F, [depth_val])
                            
                            final_color = color_tuple
                            if transparency > 0.0:
                                alpha_val = 1.0 - transparency
                                dest_color = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                                final_color = [s*alpha_val + d*(1-alpha_val) for s,d in zip(color_tuple, dest_color)]
                            
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.RGB8, final_color)


    @staticmethod
    def _draw_triangle_pipeline(p0, p1, p2, c0, c1, c2, uv0, uv1, uv2, z_cam0, z_cam1, z_cam2, mipmaps, default_color_tuple, transparency):
        """A full-pipeline rasterizer for a single triangle."""
        (x0,y0,z0,w0), (x1,y1,z1,w1), (x2,y2,z2,w2) = p0, p1, p2
        
        min_x, max_x = max(0, int(min(x0,x1,x2))), min(GL.render_width-1, int(max(x0,x1,x2)))
        min_y, max_y = max(0, int(min(y0,y1,y2))), min(GL.render_height-1, int(max(y0,y1,y2)))

        def edge(ax, ay, bx, by, px, py): return (px-ax)*(by-ay)-(py-ay)*(bx-ax)
        area = edge(x0,y0,x1,y1,x2,y2)
        if abs(area) < 1e-6: return
        
        mip_level = 0
        if mipmaps and all(uv is not None for uv in [uv0, uv1, uv2]):
            mip_level = GL._choose_mip_level(p0, p1, p2, uv0, uv1, uv2, mipmaps[0], len(mipmaps))

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                wA, wB, wC = edge(x1,y1,x2,y2,x,y), edge(x2,y2,x0,y0,x,y), edge(x0,y0,x1,y1,x,y)
                is_inside = (wA >= 0 and wB >= 0 and wC >= 0) or (wA <= 0 and wB <= 0 and wC <= 0)
                
                if is_inside:
                    alpha, beta, gamma = wA/area, wB/area, wC/area
                    
                    z_interp = alpha * z0 + beta * z1 + gamma * z2
                    current_depth = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]

                    if z_interp < current_depth:
                        z_camera = 1.0 / (alpha/z_cam0 + beta/z_cam1 + gamma/z_cam2)
                        
                        source_color = default_color_tuple
                        if mipmaps and all(uv is not None for uv in [uv0, uv1, uv2]):
                            u = z_camera * (alpha*uv0[0]/z_cam0 + beta*uv1[0]/z_cam1 + gamma*uv2[0]/z_cam2)
                            v = z_camera * (alpha*uv0[1]/z_cam0 + beta*uv1[1]/z_cam1 + gamma*uv2[1]/z_cam2)
                            tex = mipmaps[mip_level]
                            h, w, _ = tex.shape
                            tex_x, tex_y = int(u*(w-1)), int((1-v)*(h-1))
                            source_color = tex[max(0,min(h-1,tex_x)), max(0,min(w-1,tex_y))][:3]

                        elif c0 is not None:
                            r = z_camera * (alpha*c0[0]/z_cam0 + beta*c1[0]/z_cam1 + gamma*c2[0]/z_cam2)
                            g = z_camera * (alpha*c0[1]/z_cam0 + beta*c1[1]/z_cam1 + gamma*c2[1]/z_cam2)
                            b = z_camera * (alpha*c0[2]/z_cam0 + beta*c1[2]/z_cam1 + gamma*c2[2]/z_cam2)
                            source_color = [int(c*255) for c in [r,g,b]]

                        if transparency > 0.0:
                            alpha_val = 1.0 - transparency
                            dest_color = gpu.GPU.read_pixel([x,y], gpu.GPU.RGB8)
                            
                            final_color = [s*alpha_val + d*(1-alpha_val) for s,d in zip(source_color, dest_color)]
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.RGB8, final_color)
                        else: # Opaque
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.RGB8, source_color)


    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        num_points = len(point) // 2

        for i in range(num_points):
            x, y = int(point[i * 2]), int(point[i * 2 + 1])

            emissive_color = colors['emissiveColor']
            gpu.GPU.draw_pixel([int(x), int(y)], gpu.GPU.RGB8, [int(emissive_color[0] * 255), int(emissive_color[1] * 255), int(emissive_color[2] * 255)])


    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        
        emissive_color = colors['emissiveColor']

        for i in range(0, len(lineSegments) - 2, 2):
            x0, y0 = int(lineSegments[i]), int(lineSegments[i + 1])
            x1, y1 = int(lineSegments[i + 2]), int(lineSegments[i + 3])

            dx, dy = abs(x1 - x0), abs(y1 - y0)
            
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            # Algoritmo de Bresenham para desenhar a linha
            while True:
                if 0 <= x0 < GL.render_width and 0 <= y0 < GL.render_height:
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, [emissive_color[0] * 255, emissive_color[1] * 255, emissive_color[2] * 255])

                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
 
    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        emissive_color = colors['emissiveColor']

        # Assumindo que o centro esta no (0,0)
        center_x, center_y = 0, 0

        top_left = [int(center_x - radius), int(center_y - radius)]
        bottom_right = [int(center_x + radius), int(center_y + radius)]

        # Iterando na bounding box do círculo
        for x in range(top_left[0], bottom_right[0] + 1):
            for y in range(top_left[1], bottom_right[1] + 1):
                dx2 = (x - center_x) ** 2
                dy2 = (y - center_y) ** 2
                
                # Testando se o ponto esta na circunferencia
                if dx2 + dy2 <= radius ** 2 and dx2 + dy2 >= (radius - 1) ** 2:
                    if 0 <= x < GL.render_width and 0 <= y < GL.render_height:
                        gpu.GPU.draw_pixel([int(x), int(y)], gpu.GPU.RGB8, [emissive_color[0] * 255, emissive_color[1] * 255, emissive_color[2] * 255])


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        
        emissive_color = colors["emissiveColor"]
        
        for i in range(0, len(vertices), 6):
            v0_x, v0_y = int(vertices[i]), int(vertices[i + 1])
            v1_x, v1_y = int(vertices[i + 2]), int(vertices[i + 3])
            v2_x, v2_y = int(vertices[i + 4]), int(vertices[i + 5])

            top_left = [min(v0_x, v1_x, v2_x), min(v0_y, v1_y, v2_y)]
            bottom_right = [max(v0_x, v1_x, v2_x), max(v0_y, v1_y, v2_y)]

            # Iterando na bounding box do triangulo
            for x in range(top_left[0], bottom_right[0] + 1):
                for y in range(top_left[1], bottom_right[1] + 1):
                    L0 = np.matrix([[x - v0_x, y - v0_y],
                                    [v1_x - v0_x, v1_y - v0_y]])
                    
                    L1 = np.matrix([[x - v1_x, y - v1_y],
                                    [v2_x - v1_x, v2_y - v1_y]])
                    
                    L2 = np.matrix([[x - v2_x, y - v2_y],
                                    [v0_x - v2_x, v0_y - v2_y]])

                    # Se todos sao verdade, o ponto esta dentro do triangulo
                    if np.linalg.det(L0) >= 0 and np.linalg.det(L1) >= 0 and np.linalg.det(L2) >= 0:
                        if 0 <= x < GL.render_width and 0 <= y < GL.render_height:
                            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [emissive_color[0] * 255, emissive_color[1] * 255, emissive_color[2] * 255])   


    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        if not point or len(point) < 9:
            return

        def transform_vertex(v):
            """Transforms a single vertex and returns all necessary data."""
            vec4 = np.array([v[0], v[1], v[2], 1.0])
            
            # --- THIS IS THE FIX ---
            # The full transformation must include the model, view, and projection matrices.
            world_pos_vec = GL.model_matrix @ vec4
            view_pos_vec = GL.view_matrix @ world_pos_vec
            clip_pos = GL.projection_matrix @ view_pos_vec
            
            world_pos = world_pos_vec[:3]
            
            w_clip = clip_pos[3] if clip_pos[3] != 0 else 1.0
            ndc_pos = clip_pos / w_clip
            
            x = int(round((ndc_pos[0] * 0.5 + 0.5) * GL.render_width))
            y = int(round((1.0 - (ndc_pos[1] * 0.5 + 0.5)) * GL.render_height))
            
            return (x, y, ndc_pos[2], w_clip, world_pos)

        def draw_filled_triangle(p0, p1, p2, world_normal):
            """Robust rasterizer with per-pixel lighting, z-buffer, and blending."""
            (x0, y0, z0, w0, world0), (x1, y1, z1, w1, world1), (x2, y2, z2, w2, world2) = p0, p1, p2
            
            min_x = max(0, int(math.floor(min(x0,x1,x2))))
            max_x = min(GL.render_width - 1, int(math.ceil(max(x0,x1,x2))))
            min_y = max(0, int(math.floor(min(y0,y1,y2))))
            max_y = min(GL.render_height - 1, int(math.ceil(max(y0,y1,y2))))

            def edge(ax, ay, bx, by, px, py): return (px-ax)*(by-ay)-(py-ay)*(bx-ax)
            area = edge(x0, y0, x1, y1, x2, y2)
            if abs(area) < 1e-6: return

            if area < 0:
                p1, p2 = p2, p1 # Swap full data tuples
                (x1, y1, z1, w1, world1) = p1
                (x2, y2, z2, w2, world2) = p2
                area = -area

            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    px, py = x + 0.5, y + 0.5
                    wA = edge(x1, y1, x2, y2, px, py)
                    wB = edge(x2, y2, x0, y0, px, py)
                    wC = edge(x0, y0, x1, y1, px, py)

                    if wA >= 0 and wB >= 0 and wC >= 0:
                        alpha, beta, gamma = wA/area, wB/area, wC/area
                        z_interp_ndc = alpha*z0 + beta*z1 + gamma*z2
                        depth_val = (z_interp_ndc + 1.0) * 0.5
                        current_depth = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]
                        
                        if depth_val < current_depth:
                            world_pos_interp = alpha*world0 + beta*world1 + gamma*world2
                            
                            lit_color_float = GL._calculate_lighting(world_normal, world_pos_interp, colors)
                            source_color = [int(c * 255) for c in lit_color_float]

                            final_color = source_color
                            if transparency > 0.0:
                                alpha_val = 1.0 - transparency
                                dest_color = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                                final_color = [s*alpha_val + d*(1-alpha_val) for s,d in zip(source_color, dest_color)]
                            
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.DEPTH_COMPONENT32F, [depth_val])
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.RGB8, final_color)

        # --- MAIN LOGIC for triangleSet ---
        transparency = colors.get("transparency", 0.0)

        for i in range(0, len(point), 9):
            v0, v1, v2 = point[i:i+3], point[i+3:i+6], point[i+6:i+9]
            
            local_normal = GL._calculate_triangle_normal(v0, v1, v2)
            try: # Use inverse transpose, the mathematically correct way
                upper_3x3 = GL.model_matrix[:3, :3]
                normal_transform = np.linalg.inv(upper_3x3).T
                world_normal = normal_transform @ np.array(local_normal)
                world_normal /= np.linalg.norm(world_normal)
            except np.linalg.LinAlgError:
                world_normal = (GL.model_matrix[:3,:3] @ np.array(local_normal))
                world_normal /= np.linalg.norm(world_normal)
            
            p0_data = transform_vertex(v0)
            p1_data = transform_vertex(v1)
            p2_data = transform_vertex(v2)
            
            draw_filled_triangle(p0_data, p1_data, p2_data, world_normal.tolist())


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Sets up the view and projection matrices based on camera properties."""
        GL.eye = np.array(position)
        eye = GL.eye
        
        # Create rotation matrix from axis-angle to find camera direction
        x, y, z, angle = orientation
        c, s = math.cos(angle), math.sin(angle)
        n = math.sqrt(x*x + y*y + z*z) or 1.0
        x, y, z = x/n, y/n, z/n
        R = np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s, 0],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s, 0],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c),   0],
            [0,               0,               0,               1]
        ])

        forward = (R @ np.array([0, 0, -1, 0]))[:3]
        up = (R @ np.array([0, 1,  0, 0]))[:3]
        target = eye + forward

        # Build LookAt (View) Matrix
        f = (target - eye)
        f /= np.linalg.norm(f)
        s_ = np.cross(f, up)
        s_ /= np.linalg.norm(s_)
        u = np.cross(s_, f)
        
        GL.view_matrix = np.array([
            [s_[0], s_[1], s_[2], -np.dot(s_, eye)],
            [u[0], u[1], u[2], -np.dot(u, eye)],
            [-f[0],-f[1],-f[2],  np.dot(f, eye)],
            [0,    0,    0,    1]
        ])
        
        # Build Perspective Projection Matrix
        aspect = GL.render_width / GL.render_height
        f_persp = 1.0 / math.tan(fieldOfView / 2)
        
        GL.projection_matrix = np.array([
            [f_persp / aspect, 0, 0, 0],
            [0, f_persp, 0, 0],
            [0, 0, (GL.far + GL.near) / (GL.near - GL.far), (2*GL.far*GL.near)/(GL.near - GL.far)],
            [0, 0, -1, 0]
        ])


    @staticmethod
    def transform_in(translation, scale, rotation):
        """Pushes the current model matrix and applies a new local transformation."""
        GL.matrix_stack.append(GL.model_matrix.copy())
        
        T = np.identity(4); T[:3, 3] = translation
        S = np.diag([scale[0], scale[1], scale[2], 1])
        
        x, y, z, angle = rotation
        c, s = math.cos(angle), math.sin(angle)
        n = math.sqrt(x*x + y*y + z*z) or 1.0
        x, y, z = x/n, y/n, z/n
        R = np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s, 0],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s, 0],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c),   0],
            [0,               0,               0,               1]
        ])
        
        local_transform = T @ R @ S
        GL.model_matrix = GL.model_matrix @ local_transform

    @staticmethod
    def transform_out():
        """Pops the model matrix to return to the parent's coordinate system."""
        if GL.matrix_stack:
            GL.model_matrix = GL.matrix_stack.pop()
        else:
            GL.model_matrix = np.identity(4)

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        if not point or not stripCount: return

        emissive_color = colors["emissiveColor"]
        color_tuple = [c * 255 for c in emissive_color]
        
        verts = [point[i:i+3] for i in range(0, len(point), 3)]
        
        full_transform = GL.projection_matrix @ GL.view_matrix @ GL.model_matrix
        view_model_transform = GL.view_matrix @ GL.model_matrix
        
        vert_idx = 0
        for count in stripCount:
            if count < 3:
                vert_idx += count
                continue

            for i in range(count - 2):
                i0, i1, i2 = vert_idx + i, vert_idx + i + 1, vert_idx + i + 2
                
                # Ensure correct winding order for strips
                v_indices = (i0, i1, i2) if i % 2 == 0 else (i0, i2, i1)
                
                v0, v1, v2 = verts[v_indices[0]], verts[v_indices[1]], verts[v_indices[2]]
                v0_h, v1_h, v2_h = np.array(v0+[1.0]), np.array(v1+[1.0]), np.array(v2+[1.0])

                z_cam0 = -(view_model_transform @ v0_h)[2]
                z_cam1 = -(view_model_transform @ v1_h)[2]
                z_cam2 = -(view_model_transform @ v2_h)[2]
                if z_cam0 < GL.near or z_cam1 < GL.near or z_cam2 < GL.near: continue

                p0_clip, p1_clip, p2_clip = full_transform@v0_h, full_transform@v1_h, full_transform@v2_h

                if p0_clip[3]==0 or p1_clip[3]==0 or p2_clip[3]==0: continue
                p0_ndc, p1_ndc, p2_ndc = p0_clip/p0_clip[3], p1_clip/p1_clip[3], p2_clip/p2_clip[3]

                sx0, sy0 = int((p0_ndc[0]+1)*0.5*GL.render_width), int((1-p0_ndc[1])*0.5*GL.render_height)
                sx1, sy1 = int((p1_ndc[0]+1)*0.5*GL.render_width), int((1-p1_ndc[1])*0.5*GL.render_height)
                sx2, sy2 = int((p2_ndc[0]+1)*0.5*GL.render_width), int((1-p2_ndc[1])*0.5*GL.render_height)

                p0_final = (sx0, sy0, p0_ndc[2], p0_clip[3])
                p1_final = (sx1, sy1, p1_ndc[2], p1_clip[3])
                p2_final = (sx2, sy2, p2_ndc[2], p2_clip[3])
                
                GL._draw_triangle_simple(p0_final, p1_final, p2_final, color_tuple, 0)

            vert_idx += count


    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        if not point or not index:
            return

        # --- SETUP ---
        transparency = colors.get("transparency", 0.0)
        verts = [point[i:i+3] for i in range(0, len(point), 3)]
        
        # --- NESTED HELPER FUNCTIONS ---
        # (These helpers are identical to the ones in the corrected triangleSet)

        def transform_vertex(v):
            """Transforms a single vertex and returns all necessary data."""
            vec4 = np.array([v[0], v[1], v[2], 1.0])
            world_pos = (GL.model_matrix @ vec4)[:3]
            clip_pos = GL.projection_matrix @ GL.view_matrix @ GL.model_matrix @ vec4
            w_clip = clip_pos[3] if clip_pos[3] != 0 else 1.0
            ndc_pos = clip_pos / w_clip
            x = int(round((ndc_pos[0] * 0.5 + 0.5) * GL.render_width))
            y = int(round((1.0 - (ndc_pos[1] * 0.5 + 0.5)) * GL.render_height))
            return (x, y, ndc_pos[2], w_clip, world_pos)

        def draw_filled_triangle(p0, p1, p2, world_normal):
            """Robust rasterizer with per-pixel lighting, z-buffer, and blending."""
            (x0, y0, z0, w0, world0), (x1, y1, z1, w1, world1), (x2, y2, z2, w2, world2) = p0, p1, p2
            min_x = max(0, int(math.floor(min(x0,x1,x2))))
            max_x = min(GL.render_width - 1, int(math.ceil(max(x0,x1,x2))))
            min_y = max(0, int(math.floor(min(y0,y1,y2))))
            max_y = min(GL.render_height - 1, int(math.ceil(max(y0,y1,y2))))
            def edge(ax, ay, bx, by, px, py): return (px-ax)*(by-ay)-(py-ay)*(bx-ax)
            area = edge(x0, y0, x1, y1, x2, y2)
            if abs(area) < 1e-6: return
            if area < 0:
                p1, p2 = p2, p1
                (x1, y1, z1, w1, world1) = p1; (x2, y2, z2, w2, world2) = p2
                area = -area
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    px, py = x + 0.5, y + 0.5
                    wA = edge(x1, y1, x2, y2, px, py); wB = edge(x2, y2, x0, y0, px, py); wC = edge(x0, y0, x1, y1, px, py)
                    if wA >= 0 and wB >= 0 and wC >= 0:
                        alpha, beta, gamma = wA/area, wB/area, wC/area
                        z_interp_ndc = alpha*z0 + beta*z1 + gamma*z2
                        depth_val = (z_interp_ndc + 1.0) * 0.5
                        current_depth = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]
                        if depth_val < current_depth:
                            world_pos_interp = alpha*world0 + beta*world1 + gamma*world2
                            lit_color_float = GL._calculate_lighting(world_normal, world_pos_interp, colors)
                            source_color = [int(c * 255) for c in lit_color_float]
                            final_color = source_color
                            if transparency > 0.0:
                                alpha_val = 1.0 - transparency
                                dest_color = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                                final_color = [s*alpha_val + d*(1-alpha_val) for s,d in zip(source_color, dest_color)]
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.DEPTH_COMPONENT32F, [depth_val])
                            gpu.GPU.draw_pixel([x,y], gpu.GPU.RGB8, final_color)

        # --- MAIN LOGIC for indexedTriangleStripSet ---
        current_strip = []
        for idx in index:
            if idx == -1:
                if len(current_strip) >= 3:
                    for i in range(len(current_strip) - 2):
                        i0, i1, i2 = current_strip[i], current_strip[i+1], current_strip[i+2]
                        
                        # Ensure correct winding order for strips
                        v_indices = (i0, i1, i2) if i % 2 == 0 else (i0, i2, i1)
                        
                        v0, v1, v2 = verts[v_indices[0]], verts[v_indices[1]], verts[v_indices[2]]
                        
                        local_normal = GL._calculate_triangle_normal(v0, v1, v2)
                        try:
                            upper_3x3 = GL.model_matrix[:3, :3]
                            normal_transform = np.linalg.inv(upper_3x3).T
                            world_normal = normal_transform @ np.array(local_normal)
                            world_normal /= np.linalg.norm(world_normal)
                        except np.linalg.LinAlgError:
                            world_normal = (GL.model_matrix[:3,:3] @ np.array(local_normal))
                            world_normal /= np.linalg.norm(world_normal)
                            
                        p0_data = transform_vertex(v0)
                        p1_data = transform_vertex(v1)
                        p2_data = transform_vertex(v2)

                        draw_filled_triangle(p0_data, p1_data, p2_data, world_normal.tolist())
                current_strip = []
            else:
                current_strip.append(idx)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        if not coord or not coordIndex: return

        verts = [coord[i:i+3] for i in range(0, len(coord), 3)]
        faces = [[]]
        for idx in coordIndex:
            if idx == -1: 
                if faces[-1]: faces.append([])
            else: faces[-1].append(idx)
        if not faces[-1]: faces.pop()

        vert_colors = [color[i:i+3] for i in range(0, len(color), 3)] if colorPerVertex and color else None
        vert_tex_coords = [texCoord[i:i+2] for i in range(0, len(texCoord), 2)] if texCoord else None
        
        transparency = colors.get("transparency", 0.0)
        emissive_color = colors["emissiveColor"]
        default_color_tuple = [c * 255 for c in emissive_color]

        mipmaps = None
        if current_texture and current_texture[0] and vert_tex_coords:
            try:
                base_img = gpu.GPU.load_texture(current_texture[0])
                mipmaps = [base_img]
                while mipmaps[-1].shape[0] > 1 or mipmaps[-1].shape[1] > 1:
                    prev, (h,w,chans) = mipmaps[-1], mipmaps[-1].shape
                    new_h, new_w = max(1, h//2), max(1, w//2)
                    new_level = np.zeros((new_h, new_w, chans), dtype=np.uint8)
                    for y in range(new_h):
                        for x in range(new_w):
                            new_level[y, x] = prev[y*2:y*2+2, x*2:x*2+2].mean(axis=(0,1))
                    mipmaps.append(new_level)
            except Exception as e:
                print(f"Could not load/process texture: {e}")

        full_transform = GL.projection_matrix @ GL.view_matrix @ GL.model_matrix
        view_model_transform = GL.view_matrix @ GL.model_matrix
        
        for face in faces:
            p0_idx = face[0]
            for i in range(1, len(face) - 1):
                p1_idx, p2_idx = face[i], face[i+1]
                
                v0, v1, v2 = verts[p0_idx], verts[p1_idx], verts[p2_idx]
                v0_h, v1_h, v2_h = np.array(v0+[1.0]), np.array(v1+[1.0]), np.array(v2+[1.0])
                
                z_cam0 = -(view_model_transform @ v0_h)[2]
                z_cam1 = -(view_model_transform @ v1_h)[2]
                z_cam2 = -(view_model_transform @ v2_h)[2]
                if z_cam0 < GL.near or z_cam1 < GL.near or z_cam2 < GL.near: continue

                p0_clip, p1_clip, p2_clip = full_transform@v0_h, full_transform@v1_h, full_transform@v2_h
                
                if p0_clip[3]==0 or p1_clip[3]==0 or p2_clip[3]==0: continue
                p0_ndc, p1_ndc, p2_ndc = p0_clip/p0_clip[3], p1_clip/p1_clip[3], p2_clip/p2_clip[3]
                
                sx0, sy0 = int((p0_ndc[0]+1)*0.5*GL.render_width), int((1-p0_ndc[1])*0.5*GL.render_height)
                sx1, sy1 = int((p1_ndc[0]+1)*0.5*GL.render_width), int((1-p1_ndc[1])*0.5*GL.render_height)
                sx2, sy2 = int((p2_ndc[0]+1)*0.5*GL.render_width), int((1-p2_ndc[1])*0.5*GL.render_height)
                
                p0_final = (sx0, sy0, p0_ndc[2], p0_clip[3])
                p1_final = (sx1, sy1, p1_ndc[2], p1_clip[3])
                p2_final = (sx2, sy2, p2_ndc[2], p2_clip[3])

                c0,c1,c2 = (None,None,None)
                if vert_colors:
                    c0,c1,c2 = vert_colors[p0_idx], vert_colors[p1_idx], vert_colors[p2_idx]
                uv0,uv1,uv2 = (None,None,None)
                if vert_tex_coords:
                    uv0,uv1,uv2 = vert_tex_coords[p0_idx], vert_tex_coords[p1_idx], vert_tex_coords[p2_idx]
                
                GL._draw_triangle_pipeline(p0_final, p1_final, p2_final, c0, c1, c2, uv0, uv1, uv2, z_cam0, z_cam1, z_cam2, mipmaps, default_color_tuple, transparency)


    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        if headlight:
            headlight_info = {
                'type': 'directional', # A headlight acts like a directional light
                'color': [1.0, 1.0, 1.0],
                'intensity': 1.0,
                'ambientIntensity': 0.0,
                'direction': [0.0, 0.0, -1.0] # Shines straight ahead in view space
            }
            GL.lights.append(headlight_info)

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        norm_direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else direction

        GL.lights.append({
            'type': 'directional',
            'ambientIntensity': ambientIntensity,
            'color': color,
            'intensity': intensity,
            'direction': norm_direction.tolist()
        })

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
