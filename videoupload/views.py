import os
from django.conf import settings
from django.shortcuts import render
from videoupload.forms import VideoUploadForm
from vivit.vivit import vivit
from django.http import JsonResponse


def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            video_name = video.name
            media_directory = os.path.join(settings.MEDIA_ROOT, 'videos')
            os.makedirs(media_directory, exist_ok=True)

            video_path = os.path.join(media_directory, video_name)

            with open(video_path, 'wb') as destination:
                for chunk in video.chunks():
                    destination.write(chunk)

            result = vivit(video_path)
            os.remove(video_path)
            return JsonResponse({"result": result})
    else:
        form = VideoUploadForm()
    return render(request, 'upload.html', {'form': form})
