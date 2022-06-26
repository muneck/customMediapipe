#include <vector>

#include "hand_tracking_detect.h"
#include "hand_gesture_recognition.h"
#include "hand_up_hand_down_detect.h"


GoogleMediapipeHandTrackingDetect::HandTrackingDetect::HandTrackingDetect()
{
	m_bIsInit = false;
	m_bIsRelease = false;
	m_kInputStream = "input_video";
	m_kOutputStream = "output_video";
	m_kWindowName = "MediaPipe";
	m_kOutputLandmarks = "landmarks";
	m_LandmarksCallBackFunc = nullptr;
	m_GestureResultCallBackFunc = nullptr;
}

GoogleMediapipeHandTrackingDetect::HandTrackingDetect::~HandTrackingDetect()
{
	if (!m_bIsRelease)
	{
		Release();
	}
}

int GoogleMediapipeHandTrackingDetect::HandTrackingDetect::InitGraph(const char* model_path)
{
	absl::Status run_status = Mediapipe_InitGraph(model_path);
	if (!run_status.ok()) 
	{
		return 0;
	}
	m_bIsInit = true;
	return  1;
}


int GoogleMediapipeHandTrackingDetect::HandTrackingDetect::RegisterLandmarksCallback(LandmarksCallBack func)
{
	if (func != nullptr)
	{
		m_LandmarksCallBackFunc = func;
		return 1;
	}

	return 0;
}

int GoogleMediapipeHandTrackingDetect::HandTrackingDetect::RegisterGestureResultCallBack(GestureResultCallBack func)
{
	if (func != nullptr)
	{
		m_GestureResultCallBackFunc = func;
		return 1;
	}

	return 0;
}



int GoogleMediapipeHandTrackingDetect::HandTrackingDetect::DetectFrame_Direct(int image_width, int image_height, void* image_data, Gesture& gesture_result)
{
	if (!m_bIsInit)
		return 0;

	absl::Status run_status = Mediapipe_RunMPPGraph_Direct(image_width, image_height, image_data, gesture_result);
	if (!run_status.ok()) {
		return 0;
	}
	return 1;
}


int GoogleMediapipeHandTrackingDetect::HandTrackingDetect::Release()
{
	absl::Status run_status = Mediapipe_ReleaseGraph();
	if (!run_status.ok()) {
		return 0;
	}
	m_bIsRelease = true;
	return 1;
}


absl::Status GoogleMediapipeHandTrackingDetect::HandTrackingDetect::Mediapipe_InitGraph(const char* model_path)
{
	std::string calculator_graph_config_contents;
	MP_RETURN_IF_ERROR(mediapipe::file::GetContents(model_path, &calculator_graph_config_contents));

	mediapipe::CalculatorGraphConfig config =
		mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
			calculator_graph_config_contents);

	MP_RETURN_IF_ERROR(m_Graph.Initialize(config));

	// 添加video输出流
	auto sop = m_Graph.AddOutputStreamPoller(m_kOutputStream);
	assert(sop.ok());
	m_pPoller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));

	// 添加landmarks输出流
	mediapipe::StatusOrPoller sop_landmark = m_Graph.AddOutputStreamPoller(m_kOutputLandmarks);
	assert(sop_landmark.ok());
	m_pPoller_landmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop_landmark.value()));

	MP_RETURN_IF_ERROR(m_Graph.StartRun({}));

	return absl::OkStatus();
}


absl::Status GoogleMediapipeHandTrackingDetect::HandTrackingDetect::Mediapipe_RunMPPGraph_Direct(int image_width, int image_height, void* image_data, Gesture& gesture_result)
{
	// 构造cv::Mat对象
	cv::Mat camera_frame(cv::Size(image_width, image_height), CV_8UC3, (uchar*)image_data);
	cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGB);
	cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
	//std::cout << "图片构建完成" << std::endl;

	// Wrap Mat into an ImageFrame.
	auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
		mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
		mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
	camera_frame.copyTo(input_frame_mat);
	//std::cout << "Wrap Mat into an ImageFrame." << std::endl;

	// Send image packet into the graph.
	size_t frame_timestamp_us =
		(double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

	MP_RETURN_IF_ERROR(m_Graph.AddPacketToInputStream(
		m_kInputStream, mediapipe::Adopt(input_frame.release())
		.At(mediapipe::Timestamp(frame_timestamp_us))));
	//std::cout << "Send image packet into the graph." << std::endl;


	// Get the graph result packet, or stop if that fails.
	mediapipe::Packet packet;
	mediapipe::Packet packet_landmarks;
	if (!m_pPoller->Next(&packet))
		return absl::OkStatus();

	if (m_pPoller_landmarks->QueueSize() > 0)
	{
		if (m_pPoller_landmarks->Next(&packet_landmarks))
		{

			std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

			mediapipe::NormalizedLandmarkList single_hand_NormalizedLandmarkList = output_landmarks[0];

			std::vector<PoseInfo> singleHandGestureInfo;
			singleHandGestureInfo.clear();

			for (int i = 0; i < single_hand_NormalizedLandmarkList.landmark_size(); ++i)
			{
				PoseInfo info;
				const mediapipe::NormalizedLandmark landmark = single_hand_NormalizedLandmarkList.landmark(i);
				info.x = landmark.x() * camera_frame.cols;
				info.y = landmark.y() * camera_frame.rows; 
				info.z = landmark.z() * camera_frame.cols;
				singleHandGestureInfo.push_back(info);
			}

			// 检测姿势
			HandGestureRecognition handGestureRecognition;
			Gesture gesture_recognition_result = handGestureRecognition.GestureRecognition(singleHandGestureInfo);
			gesture_result = gesture_recognition_result;
		}
	}

	return absl::OkStatus();
}




absl::Status GoogleMediapipeHandTrackingDetect::HandTrackingDetect::Mediapipe_ReleaseGraph()
{
	MP_RETURN_IF_ERROR(m_Graph.CloseInputStream(m_kInputStream));
	return m_Graph.WaitUntilDone();
}
