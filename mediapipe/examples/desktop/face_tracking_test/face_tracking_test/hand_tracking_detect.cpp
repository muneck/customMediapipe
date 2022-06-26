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
	m_kOutputLandmarks = "multi_face_landmarks";
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
	std::cout << "log:"<<std::endl;

	//std::string calculator_graph_config_contents;
	//MP_RETURN_IF_ERROR(mediapipe::file::GetContents(model_path, &calculator_graph_config_contents));

	//mediapipe::CalculatorGraphConfig config =
	//	mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
	//		calculator_graph_config_contents);

	//std::cout << "initialize..." << std::endl;

	//MP_RETURN_IF_ERROR(m_Graph.Initialize(config));
	std::cout << "prepare config:" << std::endl;

	std::cout << graphConfig << std::endl;

	mediapipe::CalculatorGraphConfig config =
		mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
			graphConfig);

	std::cout << "init:" << std::endl;

	MP_RETURN_IF_ERROR(m_Graph.Initialize(config));

	std::cout << "add stream..." << std::endl;

	// 添加输出流

	//ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller face_decection_poller,
	//	m_Graph.AddOutputStreamPoller("DETECTIONS"));
	//ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller presence_poller,
	//	m_Graph.AddOutputStreamPoller("landmark_presence"));
	ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
		m_Graph.AddOutputStreamPoller("multi_face_landmarks"));


	//face_detection_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
	//	std::move(face_decection_poller));
	//presence_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
	//	std::move(presence_poller));
	landmarks_poller_ptr = std::make_unique<mediapipe::OutputStreamPoller>(
		std::move(landmarks_poller));

	std::cout << "start run..." << std::endl;

	MP_RETURN_IF_ERROR(m_Graph.StartRun({}));

	return absl::OkStatus();
}


absl::Status GoogleMediapipeHandTrackingDetect::HandTrackingDetect::Mediapipe_RunMPPGraph_Direct(int image_width, int image_height, void* image_data, Gesture& gesture_result)
{
	// 构造cv::Mat对象
	cv::Mat camera_frame(cv::Size(image_width, image_height), CV_8UC3, (uchar*)image_data);
	cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGB);
	cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

	// Wrap Mat into an ImageFrame.
	std::cout << "Wrap Mat into an ImageFrame..." << std::endl;
	auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
		mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
		mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
	camera_frame.copyTo(input_frame_mat);

	// Send image packet into the graph.
	std::cout << "Send image packet into the graph..." << std::endl;
	size_t frame_timestamp_us =
		(double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

	//MP_RETURN_IF_ERROR(m_Graph.AddPacketToInputStream(
	//	m_kInputStream, mediapipe::Adopt(input_frame.release())
	//	.At(mediapipe::Timestamp(frame_timestamp_us))));

	m_Graph.AddPacketToInputStream(
		m_kInputStream, mediapipe::Adopt(input_frame.release())
		.At(mediapipe::Timestamp(frame_timestamp_us)));


	// Get the graph result packet, or stop if that fails.
	mediapipe::Packet packet;
	mediapipe::Packet packet_landmarks;
	mediapipe::Packet packet_precence;


	//if (!m_pPoller->Next(&packet))
	//	return absl::OkStatus();
	std::cout << "Get the graph result packet..." << std::endl;

	if (landmarks_poller_ptr->QueueSize() > 0)
	{
		if (landmarks_poller_ptr->Next(&packet_landmarks))
		{
			//auto if_present = packet_precence.Get<bool>();
			//if(if_present)
			//{
			//	std::cout << "precence"<< std::endl;
			//}
			//else
			//{
			//	std::cout << "not precence" << std::endl;

			//}

			std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

			std::cout << "face_number:" << output_landmarks.size() << std::endl;
			mediapipe::NormalizedLandmarkList face_NormalizedLandmarkList = output_landmarks[0];
			std::cout << "landmarks_number:" << face_NormalizedLandmarkList.landmark_size() << std::endl;
			std::vector<PoseInfo> singleHandGestureInfo;
			singleHandGestureInfo.clear();
			Gesture gesture_recognition_result;

			gesture_recognition_result.turn = face_NormalizedLandmarkList.landmark(33).z() - face_NormalizedLandmarkList.landmark(263).z();
			gesture_recognition_result.tilt = face_NormalizedLandmarkList.landmark(33).y() - face_NormalizedLandmarkList.landmark(263).y();
			gesture_recognition_result.nod = face_NormalizedLandmarkList.landmark(1).z() - face_NormalizedLandmarkList.landmark(168).z();

			// 检测姿势
			gesture_result = gesture_recognition_result;
		}
		else
		{
			std::cout << "landmarks_poller_ptr->next()" << std::endl;

		}
	}
	else
	{
		std::cout << "QueueSize empty." << std::endl;
	}

	return absl::OkStatus();
}




absl::Status GoogleMediapipeHandTrackingDetect::HandTrackingDetect::Mediapipe_ReleaseGraph()
{
	MP_RETURN_IF_ERROR(m_Graph.CloseInputStream(m_kInputStream));
	return m_Graph.WaitUntilDone();
}


