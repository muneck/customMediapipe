#ifndef HAND_GESTURE_RECOGNITION_H
#define HAND_GESTURE_RECOGNITION_H

#include "hand_tracking_data.h"

#include <vector>


namespace GoogleMediapipeHandTrackingDetect {

	class HandGestureRecognition
	{
	public:
		HandGestureRecognition();
		virtual~HandGestureRecognition();

	public:
		Gesture GestureRecognition(const std::vector<PoseInfo>& single_hand_joint_vector);

	};

}


#endif // !HAND_GESTURE_RECOGNITION_H
