
from django.urls import path
from . import views

urlpatterns = [
    path("", views.cps_dashboard, name = 'dashboard'),
    path('prediction/', views.predictor, name='prediction'),
    path('prediction/result/', views.formInfo, name = 'result'),
    path('feature/', views.feature_analysis, name = 'feature'),
    path('batch-predict/', views.batch_predictor, name='batch-predict'),
    path('exploration/', views.data_exploration, name='exploration'),
]
