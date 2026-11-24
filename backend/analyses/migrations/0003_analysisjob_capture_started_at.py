from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analyses", "0002_analysisjob_overlay_video"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysisjob",
            name="capture_started_at",
            field=models.DateTimeField(blank=True, null=True, verbose_name="녹화 시작 시각"),
        ),
    ]
