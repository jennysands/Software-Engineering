# Generated by Django 4.1.3 on 2022-11-26 19:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SE_Site', '0003_birdfile_delete_birdaudio_delete_birdimage'),
    ]

    operations = [
        migrations.CreateModel(
            name='BirdImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bird_image_location', models.FileField(upload_to='')),
            ],
        ),
        migrations.DeleteModel(
            name='BirdFile',
        ),
    ]
