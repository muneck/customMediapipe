#ifndef HAND_TRACKING_DETECT_H
#define HAND_TRACKING_DETECT_H

#include <cstdlib>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_replace.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

#include "hand_tracking_data.h"


namespace GoogleMediapipeHandTrackingDetect {

	typedef void(*LandmarksCallBack)(int image_index,PoseInfo* infos, int count);
	typedef void(*GestureResultCallBack)(int image_index, int* recogn_result, int count);

	class HandTrackingDetect
	{
	public:
		HandTrackingDetect();
		virtual ~HandTrackingDetect();

	public:
		int InitGraph(const char* model_path);
		int RegisterLandmarksCallback(LandmarksCallBack func);
		int RegisterGestureResultCallBack(GestureResultCallBack func);
		int DetectFrame_Direct(int image_width, int image_height, void* image_data,Gesture& gesture_result);
		int Release();

	private:
		absl::Status Mediapipe_InitGraph(const char* model_path);

		absl::Status Mediapipe_RunMPPGraph_Direct(int image_width, int image_height, void* image_data, Gesture& gesture_result);

		absl::Status Mediapipe_ReleaseGraph();

		

	private:
		bool m_bIsInit;
		bool m_bIsRelease;

		const char* m_kInputStream;
		const char* m_kOutputStream;
		const char* m_kWindowName;
		const char* m_kOutputLandmarks;

		LandmarksCallBack m_LandmarksCallBackFunc;
		GestureResultCallBack m_GestureResultCallBackFunc;

		mediapipe::CalculatorGraph m_Graph;

		std::unique_ptr<mediapipe::OutputStreamPoller> m_pPoller;
		std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_ptr;
		std::unique_ptr<mediapipe::OutputStreamPoller> presence_poller_ptr;
		const std::string graphConfig = R"pb(
# MediaPipe graph that performs face mesh with TensorFlow Lite on CPU.

# Input image. (ImageFrame)
input_stream: "input_video"

# Output image with rendered results. (ImageFrame)
output_stream: "output_video"
# Collection of detected/processed faces, each represented as a list of
# landmarks. (std::vector<NormalizedLandmarkList>)
output_stream: "multi_face_landmarks"


# Throttles the images flowing downstream for flow control. It passes through
# the very first incoming image unaltered, and waits for downstream nodes
# (calculators and subgraphs) in the graph to finish their tasks before it
# passes through another image. All images that come in while waiting are
# dropped, limiting the number of in-flight images in most part of the graph to
# 1. This prevents the downstream nodes from queuing up incoming images and data
# excessively, which leads to increased latency and memory usage, unwanted in
# real-time mobile applications. It also eliminates unnecessarily computation,
# e.g., the output produced by a node may get dropped downstream if the
# subsequent nodes are still busy processing previous inputs.
node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:output_video"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

# Defines side packets for further use in the graph.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:num_faces"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: 1 }
    }
  }
}

# Subgraph that detects faces and corresponding landmarks.
node {
  calculator: "FaceLandmarkFrontCpu"
  input_stream: "IMAGE:throttled_input_video"
  input_side_packet: "NUM_FACES:num_faces"
  output_stream: "LANDMARKS:multi_face_landmarks"
  output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
  output_stream: "DETECTIONS:face_detections"
  output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
}

# Subgraph that renders face-landmark annotation onto the input image.
node {
  calculator: "FaceRendererCpu"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "LANDMARKS:multi_face_landmarks"
  input_stream: "NORM_RECTS:face_rects_from_landmarks"
  input_stream: "DETECTIONS:face_detections"
  output_stream: "IMAGE:output_video"
}

)pb";

	};

}


#endif // HAND_TRACKING_DETECT_H
 