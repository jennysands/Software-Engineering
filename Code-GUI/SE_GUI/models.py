from django.db import models

# Create your models here.

class BirdImage(models.Model):
    bird_image_location = models.ImageField(upload_to='images/')
    
class BirdAudio(models.Model):
    bird_audio_location = models.FileField(upload_to='audio/')
