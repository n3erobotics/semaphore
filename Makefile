all:	red_object_tracking

red_object_tracking: SemaphoreTrigger.cpp
	g++ SemaphoreTrigger.cpp -o SemaphoreTrigger `pkg-config opencv cvblob --cflags --libs`
