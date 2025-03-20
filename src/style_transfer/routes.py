from flask import Blueprint, request, jsonify, send_file
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from .models import gram_matrix, NormalizationLayer, ContentLossLayer, StyleLossLayer, TotalVariationLoss, Compiler, Trainer
from .utils import ImageLoader

style_transfer_bp = Blueprint('style_transfer', __name__)


# Flask Route for Neural Style Transfer
@style_transfer_bp.route('/transfer', methods=['POST'])
def style_transfer():
    try:
        # Get files from request
        content_file = request.files['content_image']
        style_file = request.files['style_image']

        # Load images
        contentImage = ImageLoader(content_file.read())
        styleImage = ImageLoader(style_file.read())

        # Load VGG19 Model
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        from torchvision.models import VGG19_Weights
        vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

        # Define Layers for Content and Style Loss
        contentLayerNames = ['conv4']
        styleLayerNames = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

        # Compile the Model
        compiler = Compiler(vgg19, contentLayerNames, styleLayerNames, device=device)
        model, contentLayers, styleLayers = compiler.compile(contentImage, styleImage, device)

        # Train the Model
        trainer = Trainer(model, contentLayers, styleLayers, device=device)
        inputImage = contentImage.clone()
        outputImage = trainer.fit(inputImage, epochs=60, alpha=1, beta=1e6, tv_weight=1e-5, device=device)

        # Convert output image to PIL Image
        outputImage = outputImage.cpu().squeeze(0)
        outputImage = transforms.ToPILImage()(outputImage)

        # Save to a BytesIO stream and return
        output_buffer = BytesIO()
        outputImage.save(output_buffer, format="JPEG")
        output_buffer.seek(0)

        return send_file(output_buffer, mimetype='image/jpeg', as_attachment=True, download_name='output.jpg')

    except Exception as e:
        return jsonify({'error': str(e)})
