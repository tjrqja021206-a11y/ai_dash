from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analyses", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysisjob",
            name="overlay_video",
            field=models.FileField(blank=True, upload_to="overlays/", verbose_name="Overlay 영상"),
        ),
    ]
