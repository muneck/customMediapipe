#include "hand_tracking_api.h"
#include "hand_tracking_detect.h"

using namespace GoogleMediapipeHandTrackingDetect;

HandTrackingDetect m_HandTrackingDetect;

EXPORT_API int Mediapipe_Hand_Tracking_Init(const char* model_path)
{
	return m_HandTrackingDetect.InitGraph(model_path);
}


EXPORT_API int Mediapipe_Hand_Tracking_Reigeter_Landmarks_Callback(LandmarksCallBack func)
{
	return m_HandTrackingDetect.RegisterLandmarksCallback(func);
}

EXPORT_API int Mediapipe_Hand_Tracking_Register_Gesture_Result_Callback(GestureResultCallBack func)
{
	return m_HandTrackingDetect.RegisterGestureResultCallBack(func);
}


EXPORT_API int Mediapipe_Hand_Tracking_Detect_Frame_Direct(int image_width, int image_height, void* image_data, Gesture& gesture_result)
{
	return m_HandTrackingDetect.DetectFrame_Direct(image_width, image_height, image_data, gesture_result);
}


EXPORT_API int Mediapipe_Hand_Tracking_Release()
{
	return m_HandTrackingDetect.Release();
}