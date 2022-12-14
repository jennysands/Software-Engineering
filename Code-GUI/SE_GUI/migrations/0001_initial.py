# Generated by Django 4.1.3 on 2022-11-27 19:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BirdAudio',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bird_audio_location', models.FileField(upload_to='audio/')),
            ],
        ),
        migrations.CreateModel(
            name='BirdImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bird_image_location', models.ImageField(upload_to='images/')),
            ],
        ),
    ]
