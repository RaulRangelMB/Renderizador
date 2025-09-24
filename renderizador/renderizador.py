#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Renderizador X3D.

Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
Disciplina: Computação Gráfica
Data: 28 de Agosto de 2020
"""

import os           # Para rotinas do sistema operacional
import argparse     # Para tratar os parâmetros da linha de comando

import gl           # Recupera rotinas de suporte ao X3D

import interface    # Janela de visualização baseada no Matplotlib
import gpu          # Simula os recursos de uma GPU

import x3d          # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import scenegraph   # Imprime o grafo de cena no console

import numpy as np

LARGURA = 60  # Valor padrão para largura da tela
ALTURA = 40   # Valor padrão para altura da tela


class Renderizador:
    """Realiza a renderização da cena informada."""

    def __init__(self):
        """Definindo valores padrão."""
        self.width = LARGURA
        self.height = ALTURA
        self.x3d_file = ""
        self.image_file = "tela.png"
        self.scene = None
        self.framebuffers = {}
        self.supersample_scale = 2
        self.framebuffers = {}

    def setup(self):
        """Configura o sistema para a renderização."""
        # Create two FrameBuffer positions for SSAA
        fbo = gpu.GPU.gen_framebuffers(2)
        self.framebuffers["INTERNAL"] = fbo[0]
        self.framebuffers["FRONT"] = fbo[1]

        # Calculate the high-resolution rendering dimensions
        render_width = self.width * self.supersample_scale
        render_height = self.height * self.supersample_scale

        # Update the GL module's dimensions to match the high-resolution buffer.
        # This ensures all drawing calculations operate in the correct coordinate space.
        gl.GL.width = render_width
        gl.GL.height = render_height
        
        # Configure and allocate the high-resolution INTERNAL buffer for drawing
        gpu.GPU.framebuffer_storage(
            self.framebuffers["INTERNAL"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            render_width,
            render_height
        )
        gpu.GPU.framebuffer_storage(
            self.framebuffers["INTERNAL"],
            gpu.GPU.DEPTH_ATTACHMENT,
            gpu.GPU.DEPTH_COMPONENT32F,
            render_width,
            render_height
        )

        # Configure and allocate the final-resolution FRONT buffer for display
        gpu.GPU.framebuffer_storage(
            self.framebuffers["FRONT"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width,
            self.height
        )

        gpu.GPU.clear_color([0, 0, 0])
        
        self.scene.viewport(width=render_width, height=render_height)

    def pre(self):
        """Rotinas pré renderização."""

        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["INTERNAL"])

        # Limpa o frame buffers atual
        gpu.GPU.clear_buffer()

        gpu.GPU.clear_depth(-1)

        # Recursos que podem ser úteis:
        # Define o valor do pixel no framebuffer: draw_pixel(coord, mode, data)
        # Retorna o valor do pixel no framebuffer: read_pixel(coord, mode)

    def pos(self):
        """Rotinas pós renderização."""
        # This is where the downsampling from the internal buffer to the front buffer happens.
        
        # Get the color data from the high-resolution buffer
        supersampled_buffer = gpu.GPU.frame_buffer[self.framebuffers["INTERNAL"]].color
        # Get the destination buffer's color data array
        final_buffer = gpu.GPU.frame_buffer[self.framebuffers["FRONT"]].color
        
        scale = self.supersample_scale

        # Efficiently downsample using numpy array reshaping and mean calculation
        # This averages every 2x2 block of pixels into a single pixel
        reshaped = supersampled_buffer.reshape(self.height, scale, self.width, scale, 3)
        final_image = reshaped.mean(axis=(1, 3)).astype(final_buffer.dtype)
        
        # Directly copy the final downsampled image into the FRONT Framebuffer's color array
        final_buffer[:] = final_image
        
        # Ensure the FRONT buffer is set for reading by the display window
        gpu.GPU.bind_framebuffer(gpu.GPU.READ_FRAMEBUFFER, self.framebuffers["FRONT"])

    def mapping(self):
        """Mapeamento de funções para as rotinas de renderização."""
        # Rotinas encapsuladas na classe GL (Graphics Library)
        x3d.X3D.renderer["Polypoint2D"] = gl.GL.polypoint2D
        x3d.X3D.renderer["Polyline2D"] = gl.GL.polyline2D
        x3d.X3D.renderer["Circle2D"] = gl.GL.circle2D
        x3d.X3D.renderer["TriangleSet2D"] = gl.GL.triangleSet2D
        x3d.X3D.renderer["TriangleSet"] = gl.GL.triangleSet
        x3d.X3D.renderer["Viewpoint"] = gl.GL.viewpoint
        x3d.X3D.renderer["Transform_in"] = gl.GL.transform_in
        x3d.X3D.renderer["Transform_out"] = gl.GL.transform_out
        x3d.X3D.renderer["TriangleStripSet"] = gl.GL.triangleStripSet
        x3d.X3D.renderer["IndexedTriangleStripSet"] = gl.GL.indexedTriangleStripSet
        x3d.X3D.renderer["IndexedFaceSet"] = gl.GL.indexedFaceSet
        x3d.X3D.renderer["Box"] = gl.GL.box
        x3d.X3D.renderer["Sphere"] = gl.GL.sphere
        x3d.X3D.renderer["Cone"] = gl.GL.cone
        x3d.X3D.renderer["Cylinder"] = gl.GL.cylinder
        x3d.X3D.renderer["NavigationInfo"] = gl.GL.navigationInfo
        x3d.X3D.renderer["DirectionalLight"] = gl.GL.directionalLight
        x3d.X3D.renderer["PointLight"] = gl.GL.pointLight
        x3d.X3D.renderer["Fog"] = gl.GL.fog
        x3d.X3D.renderer["TimeSensor"] = gl.GL.timeSensor
        x3d.X3D.renderer["SplinePositionInterpolator"] = gl.GL.splinePositionInterpolator
        x3d.X3D.renderer["OrientationInterpolator"] = gl.GL.orientationInterpolator

    def render(self):
        """Laço principal de renderização."""
        self.pre()  # executa rotina pré renderização
        self.scene.render()  # faz o traversal no grafo de cena
        self.pos()  # executa rotina pós renderização
        return gpu.GPU.get_frame_buffer()

    def main(self):
        """Executa a renderização."""
        # Tratando entrada de parâmetro
        parser = argparse.ArgumentParser(add_help=False)   # parser para linha de comando
        parser.add_argument("-i", "--input", help="arquivo X3D de entrada")
        parser.add_argument("-o", "--output", help="arquivo 2D de saída (imagem)")
        parser.add_argument("-w", "--width", help="resolução horizonta", type=int)
        parser.add_argument("-h", "--height", help="resolução vertical", type=int)
        parser.add_argument("-g", "--graph", help="imprime o grafo de cena", action='store_true')
        parser.add_argument("-p", "--pause", help="começa simulação em pausa", action='store_true')
        parser.add_argument("-q", "--quiet", help="não exibe janela", action='store_true')
        args = parser.parse_args() # parse the arguments
        if args.input:
            self.x3d_file = args.input
        if args.output:
            self.image_file = args.output
        if args.width:
            self.width = args.width
        if args.height:
            self.height = args.height

        path = os.path.dirname(os.path.abspath(self.x3d_file))

        # Iniciando simulação de GPU
        gpu.GPU(self.image_file, path)

        # Abre arquivo X3D
        self.scene = x3d.X3D(self.x3d_file)

        # Iniciando Biblioteca Gráfica
        gl.GL.setup(
            self.width,
            self.height,
            near=0.01,
            far=1000
        )

        # Funções que irão fazer o rendering
        self.mapping()

        # Se no modo silencioso não configurar janela de visualização
        if not args.quiet:
            window = interface.Interface(self.width, self.height, self.x3d_file)
            self.scene.set_preview(window)

        # carrega os dados do grafo de cena
        if self.scene:
            self.scene.parse()
            if args.graph:
                scenegraph.Graph(self.scene.root)

        # Configura o sistema para a renderização.
        self.setup()

        # Se no modo silencioso salvar imagem e não mostrar janela de visualização
        if args.quiet:
            gpu.GPU.save_image()  # Salva imagem em arquivo
        else:
            window.set_saver(gpu.GPU.save_image)  # pasa a função para salvar imagens
            window.preview(args.pause, self.render)  # mostra visualização

if __name__ == '__main__':
    renderizador = Renderizador()
    renderizador.main()
