#include "hand_gesture_recognition.h"
#include <cmath>
#include <iostream>

GoogleMediapipeHandTrackingDetect::HandGestureRecognition::HandGestureRecognition()
{

}


GoogleMediapipeHandTrackingDetect::HandGestureRecognition::~HandGestureRecognition()
{

}


Gesture GoogleMediapipeHandTrackingDetect::HandGestureRecognition::GestureRecognition(const std::vector<PoseInfo>& face_land_marks)
{
	Gesture result;

	if (face_land_marks.size() != 468)
		result.turn = -1;
		result.tilt = -1;
		result.nod = -1;
		return result;

	result.turn = face_land_marks[33].z - face_land_marks[263].z;
	result.tilt = face_land_marks[33].y - face_land_marks[263].y;
	result.nod = face_land_marks[1].z - face_land_marks[168].z;

	return result;
}
