# Generated by Django 4.1.3 on 2022-11-26 19:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SE_Site', '0007_birdaudio'),
    ]

    operations = [
        migrations.AlterField(
            model_name='birdimage',
            name='bird_image_location',
            field=models.ImageField(upload_to='media/'),
        ),
    ]
