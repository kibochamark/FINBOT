# Generated by Django 5.2 on 2025-04-11 14:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Agent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_created=True, auto_now=True)),
                ('query', models.TextField()),
                ('response', models.TextField()),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
