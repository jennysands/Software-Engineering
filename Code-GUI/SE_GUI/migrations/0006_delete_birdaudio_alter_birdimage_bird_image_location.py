# Generated by Django 4.1.3 on 2022-11-26 19:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SE_Site', '0005_birdaudio_alter_birdimage_bird_image_location'),
    ]

    operations = [
        migrations.DeleteModel(
            name='BirdAudio',
        ),
        migrations.AlterField(
            model_name='birdimage',
            name='bird_image_location',
            field=models.FileField(upload_to='media/'),
        ),
    ]
