//
// Copyright (c) 2017 Rasmus Barringer
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <nudge.h>
#include <immintrin.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define ALIGNED(n) __declspec(align(n))
#else
#define ALIGNED(n) __attribute__((aligned(n)))
#endif

#ifdef __AVX2__
#define SIMD_ALIGNED ALIGNED(32)
#else
#define SIMD_ALIGNED ALIGNED(16)
#endif

using namespace nudge;

static const float pi = 3.14159265f;

static void test_assert(bool condition, const char* format, ...) {
	if (condition)
		return;
	
	printf("Test assertion failed: ");
	
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	
	printf("\n");
	
	abort();
}

static inline void* align(Arena* arena, uintptr_t alignment) {
	uintptr_t data = (uintptr_t)arena->data;
	uintptr_t end = data + arena->size;
	uintptr_t mask = alignment-1;
	
	data = (data + mask) & ~mask;
	
	arena->data = (void*)data;
	arena->size = end - data;
	
	test_assert((intptr_t)arena->size >= 0, "Out of memory.");
	
	return arena->data;
}

static inline void* allocate(Arena* arena, uintptr_t size, uintptr_t alignment) {
	align(arena, alignment);
	
	void* data = arena->data;
	arena->data = (void*)((uintptr_t)data + size);
	arena->size -= size;
	
	test_assert((intptr_t)arena->size >= 0, "Out of memory.");
	
	return data;
}

template<class T>
static inline T* allocate_array(Arena* arena, uintptr_t count, uintptr_t alignment) {
	return static_cast<T*>(allocate(arena, sizeof(T)*count, alignment));
}

static inline float vector_distance_squared(const float a[3], const float b[3]) {
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	float dz = a[2] - b[2];
	
	return dx*dx + dy*dy + dz*dz;
}

static inline void rotate_axis_angle(float r[4], float axis_x, float axis_y, float axis_z, float angle) {
	// Setup quaternion representing rotation.
	float s = sinf(angle*0.5f);
	float c = cosf(angle*0.5f);
	
	float a[] = {
		axis_x*s,
		axis_y*s,
		axis_z*s,
		c,
	};
	
	float f = 1.0f / sqrtf(a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
	
	a[0] *= f;
	a[1] *= f;
	a[2] *= f;
	a[3] *= f;
	
	// Multiply with existing quaternion as right-hand side.
	float b[] = { r[0], r[1], r[2], r[3] };
	
	r[0] = b[0]*a[3] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1];
	r[1] = b[1]*a[3] + a[1]*b[3] + a[2]*b[0] - a[0]*b[2];
	r[2] = b[2]*a[3] + a[2]*b[3] + a[0]*b[1] - a[1]*b[0];
	r[3] = a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2];
}

static void box_box_test_case_0(Arena temporary) {
	// Test case that originally returned no contacts when there should be 4.
	printf("box_box_test_case_0\n");
	
	static const unsigned box_count = 2;
	static const unsigned body_count = 2;
	static const unsigned max_contacts = box_count*64;
	
	SIMD_ALIGNED Transform body_transforms[body_count] = {
		{
			{ 0, 0, 0 }, 0,
			{ 0.0f, 0.0f, 0.0f, 1.0f },
		},
		{
			{ -10.3300056f, -0.0125209205f, 20.0851059f }, 0,
			{ 0.0419846289f, -0.296176672f, -0.954125523f, -0.0126985274f },
		},
	};
	
	SIMD_ALIGNED uint16_t tags[box_count] = { 0, 1 };
	
	SIMD_ALIGNED BoxCollider boxes[box_count] = {
		{ { 200.0f, 10.0f, 200.0f } },
		{ { 0.5f, 1.0f, 0.25f } },
	};
	
	SIMD_ALIGNED Transform box_transforms[box_count] = {
		{
			{ 0.0f, -10.0f, 0.0f }, 0,
			{ 0.0f, 0.0f, 0.0f, 1.0f },
		},
		{
			{ 0.0f, 0.0f, 0.0f }, 1,
			{ 0.0f, 0.0f, 0.0f, 1.0f },
		},
	};
	
	SIMD_ALIGNED BodyProperties properties[body_count] = {};
	SIMD_ALIGNED BodyMomentum momentum[body_count] = {};
	SIMD_ALIGNED uint8_t idle_counters[body_count] = {};
	
	BodyData body_data = {};
	
	body_data.transforms = body_transforms;
	body_data.properties = properties;
	body_data.momentum = momentum;
	body_data.idle_counters = idle_counters;
	body_data.count = body_count;
	
	ColliderData collider_data = {};
	
	collider_data.boxes.tags = tags;
	collider_data.boxes.data = boxes;
	collider_data.boxes.transforms = box_transforms;
	collider_data.boxes.count = box_count;
	
	BodyConnections body_connections = {};
	
	SIMD_ALIGNED uint16_t body_indices[body_count] = {};
	ActiveBodies active_bodies = { body_indices, body_count };
	
	ContactData contacts = {
		allocate_array<Contact>(&temporary, max_contacts, 64),
		allocate_array<BodyPair>(&temporary, max_contacts, 64),
		allocate_array<uint64_t>(&temporary, max_contacts, 64),
		max_contacts,
	};
	
	collide(&active_bodies, &contacts, body_data, collider_data, body_connections, temporary);
	test_assert(contacts.count == 4, "Incorrect number of contacts.");
}

static void box_box_face_face_tags_0(Arena arena) {
	// Tests ensuring that contact tags are consistent when swapping the order of two face-colliding boxes with 8 edge intersections.
	printf("box_box_face_face_tags_0\n");
	
	for (unsigned i = 0; i < (1 << 12); ++i) {
		Arena temporary = arena;
		
		static const unsigned box_count = 2;
		static const unsigned body_count = 2;
		static const unsigned max_contacts = box_count*64;
		
		int ax = (i >> 0) & 3;
		int ay = (i >> 2) & 3;
		int az = (i >> 4) & 3;
		
		int bx = (i >> 6) & 3;
		int by = (i >> 8) & 3;
		int bz = (i >> 10) & 3;
		
		SIMD_ALIGNED Transform body_transforms[body_count] = {
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		SIMD_ALIGNED uint16_t tags[box_count] = { 0, 1 };
		
		SIMD_ALIGNED BoxCollider boxes[box_count] = {
			{ { 1.125f, 1.125f, 1.125f } },
			{ { 1.125f, 1.125f, 1.125f } },
		};
		
		SIMD_ALIGNED Transform box_transforms[box_count] = {
			{
				{ 0.0f, 1.0f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, -1.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		
		rotate_axis_angle(box_transforms[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		rotate_axis_angle(box_transforms[1].rotation, 1.0f, 0.0f, 0.0f, pi*1e-4f);
		
		SIMD_ALIGNED Transform box_transforms_reversed[] = {
			{
				{ 0.0f, -1.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, 1.0f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms_reversed[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 1.0f, 0.0f, 0.0f, pi*1e-4f);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		
		SIMD_ALIGNED BodyProperties properties[body_count] = {};
		SIMD_ALIGNED BodyMomentum momentum[body_count] = {};
		SIMD_ALIGNED uint8_t idle_counters[body_count] = {};
		
		BodyData body_data = {};
		
		body_data.transforms = body_transforms;
		body_data.properties = properties;
		body_data.momentum = momentum;
		body_data.idle_counters = idle_counters;
		body_data.count = body_count;
		
		ColliderData collider_data = {};
		
		collider_data.boxes.tags = tags;
		collider_data.boxes.data = boxes;
		collider_data.boxes.transforms = box_transforms;
		collider_data.boxes.count = box_count;
		
		BodyConnections body_connections = {};
		
		SIMD_ALIGNED uint16_t body_indices[body_count] = {};
		ActiveBodies active_bodies = { body_indices, body_count };
		
		ContactData contacts0 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		ContactData contacts1 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		collide(&active_bodies, &contacts0, body_data, collider_data, body_connections, temporary);
		test_assert(contacts0.count == 8, "Incorrect number of contacts.");
		
		collider_data.boxes.transforms = box_transforms_reversed;
		
		collide(&active_bodies, &contacts1, body_data, collider_data, body_connections, temporary);
		test_assert(contacts1.count == 8, "Incorrect number of contacts.");
		
		for (unsigned i = 0; i < 8; ++i) {
			const float* p0 = contacts0.data[i].position;
			
			float distance = vector_distance_squared(contacts1.data[0].position, p0);
			unsigned index1 = 0;
			
			for (unsigned j = 0; j < 8; ++j) {
				test_assert(contacts0.bodies[i].a == contacts1.bodies[j].b, "Inconsistent bodies.");
				test_assert(contacts0.bodies[i].b == contacts1.bodies[j].a, "Inconsistent bodies.");
				
				const float* p1 = contacts1.data[j].position;
				float d = vector_distance_squared(p1, p0);
				
				if (d < distance) {
					distance = d;
					index1 = j;
				}
			}
			
			test_assert(distance < 1e-3f, "No match found for contact in swapped configuration.");
			test_assert((uint32_t)contacts0.tags[i] == (((uint32_t)contacts1.tags[index1] >> 16) | ((uint32_t)contacts1.tags[index1] << 16)), "Inconsistent tags between configurations.");
		}
		
		for (unsigned i = 0; i < 8; ++i) {
			for (unsigned j = i+1; j < 8; ++j) {
				test_assert((uint32_t)contacts0.tags[i] != (uint32_t)contacts0.tags[j], "Duplicate tags within a face.");
				test_assert((uint32_t)contacts1.tags[i] != (uint32_t)contacts1.tags[j], "Duplicate tags within a face.");
			}
		}
	}
}

static void box_box_face_face_tags_1(Arena arena) {
	// Tests ensuring that contact tags are consistent when swapping the order of two face-colliding boxes with 1 vertex and 2 edge intersections.
	printf("box_box_face_face_tags_1\n");
	
	for (unsigned i = 0; i < (1 << 12); ++i) {
		Arena temporary = arena;
		
		static const unsigned box_count = 2;
		static const unsigned body_count = 2;
		static const unsigned max_contacts = box_count*64;
		
		int ax = (i >> 0) & 3;
		int ay = (i >> 2) & 3;
		int az = (i >> 4) & 3;
		
		int bx = (i >> 6) & 3;
		int by = (i >> 8) & 3;
		int bz = (i >> 10) & 3;
		
		SIMD_ALIGNED Transform body_transforms[body_count] = {
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		SIMD_ALIGNED uint16_t tags[box_count] = { 0, 1 };
		
		SIMD_ALIGNED BoxCollider boxes[box_count] = {
			{ { 1.125f, 1.125f, 1.125f } },
			{ { 1.125f, 1.125f, 1.125f } },
		};
		
		SIMD_ALIGNED Transform box_transforms[box_count] = {
			{
				{ 0.0f, 1.0f, 2.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, -1.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		
		rotate_axis_angle(box_transforms[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		rotate_axis_angle(box_transforms[1].rotation, 1.0f, 0.0f, 0.0f, -pi*1e-5f);
		
		SIMD_ALIGNED Transform box_transforms_reversed[] = {
			{
				{ 0.0f, -1.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, 1.0f, 2.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms_reversed[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 1.0f, 0.0f, 0.0f, pi*1e-5f);
		
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		
		SIMD_ALIGNED BodyProperties properties[body_count] = {};
		SIMD_ALIGNED BodyMomentum momentum[body_count] = {};
		SIMD_ALIGNED uint8_t idle_counters[body_count] = {};
		
		BodyData body_data = {};
		
		body_data.transforms = body_transforms;
		body_data.properties = properties;
		body_data.momentum = momentum;
		body_data.idle_counters = idle_counters;
		body_data.count = body_count;
		
		ColliderData collider_data = {};
		
		collider_data.boxes.tags = tags;
		collider_data.boxes.data = boxes;
		collider_data.boxes.transforms = box_transforms;
		collider_data.boxes.count = box_count;
		
		BodyConnections body_connections = {};
		
		SIMD_ALIGNED uint16_t body_indices[body_count] = {};
		ActiveBodies active_bodies = { body_indices, body_count };
		
		ContactData contacts0 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		ContactData contacts1 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		collide(&active_bodies, &contacts0, body_data, collider_data, body_connections, temporary);
		test_assert(contacts0.count == 3, "Incorrect number of contacts.");
		
		collider_data.boxes.transforms = box_transforms_reversed;
		
		collide(&active_bodies, &contacts1, body_data, collider_data, body_connections, temporary);
		test_assert(contacts1.count == 3, "Incorrect number of contacts.");
		
		for (unsigned i = 0; i < 3; ++i) {
			const float* p0 = contacts0.data[i].position;
			
			float distance = vector_distance_squared(contacts1.data[0].position, p0);
			unsigned index1 = 0;
			
			for (unsigned j = 0; j < 3; ++j) {
				test_assert(contacts0.bodies[i].a == contacts1.bodies[j].b, "Inconsistent bodies.");
				test_assert(contacts0.bodies[i].b == contacts1.bodies[j].a, "Inconsistent bodies.");
				
				const float* p1 = contacts1.data[j].position;
				float d = vector_distance_squared(p1, p0);
				
				if (d < distance) {
					distance = d;
					index1 = j;
				}
			}
			
			test_assert(distance < 1e-3f, "No match found for contact in swapped configuration.");
			test_assert((uint32_t)contacts0.tags[i] == (((uint32_t)contacts1.tags[index1] >> 16) | ((uint32_t)contacts1.tags[index1] << 16)), "Inconsistent tags between configurations.");
		}
		
		for (unsigned i = 0; i < 3; ++i) {
			for (unsigned j = i+1; j < 3; ++j) {
				test_assert((uint32_t)contacts0.tags[i] != (uint32_t)contacts0.tags[j], "Duplicate tags within a face.");
				test_assert((uint32_t)contacts1.tags[i] != (uint32_t)contacts1.tags[j], "Duplicate tags within a face.");
			}
		}
	}
}

static void box_box_edge_edge_tags(Arena arena) {
	// Tests ensuring that contact tags are consistent when swapping the order of two edge-colliding boxes.
	printf("box_box_edge_edge_tags\n");
	
	for (unsigned i = 0; i < (1 << 12); ++i) {
		Arena temporary = arena;
		
		static const unsigned box_count = 2;
		static const unsigned body_count = 2;
		static const unsigned max_contacts = box_count*64;
		
		int ax = (i >> 0) & 3;
		int ay = (i >> 2) & 3;
		int az = (i >> 4) & 3;
		
		int bx = (i >> 6) & 3;
		int by = (i >> 8) & 3;
		int bz = (i >> 10) & 3;
		
		SIMD_ALIGNED Transform body_transforms[body_count] = {
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		SIMD_ALIGNED uint16_t tags[box_count] = { 0, 1 };
		
		SIMD_ALIGNED BoxCollider boxes[box_count] = {
			{ { 1.125f, 1.125f, 1.125f } },
			{ { 1.125f, 1.125f, 1.125f } },
		};
		
		// Note that distance needs to be less than geometric because of edge-edge bias for feature stability.
		SIMD_ALIGNED Transform box_transforms[box_count] = {
			{
				{ 2.2f, 2.2f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, 0.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms[0].rotation, 1.0f, 0.0f, 0.0f, -pi*1e-4f);
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		rotate_axis_angle(box_transforms[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.25f);
		
		rotate_axis_angle(box_transforms[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		
		SIMD_ALIGNED Transform box_transforms_reversed[] = {
			{
				{ 0.0f, 0.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 2.2f, 2.2f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms_reversed[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		rotate_axis_angle(box_transforms_reversed[0].rotation, 1.0f, 0.0f, 0.0f, pi*1e-4f);
		
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		rotate_axis_angle(box_transforms_reversed[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.25f);
		
		SIMD_ALIGNED BodyProperties properties[body_count] = {};
		SIMD_ALIGNED BodyMomentum momentum[body_count] = {};
		SIMD_ALIGNED uint8_t idle_counters[body_count] = {};
		
		BodyData body_data = {};
		
		body_data.transforms = body_transforms;
		body_data.properties = properties;
		body_data.momentum = momentum;
		body_data.idle_counters = idle_counters;
		body_data.count = body_count;
		
		ColliderData collider_data = {};
		
		collider_data.boxes.tags = tags;
		collider_data.boxes.data = boxes;
		collider_data.boxes.transforms = box_transforms;
		collider_data.boxes.count = box_count;
		
		BodyConnections body_connections = {};
		
		SIMD_ALIGNED uint16_t body_indices[body_count] = {};
		ActiveBodies active_bodies = { body_indices, body_count };
		
		ContactData contacts0 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		ContactData contacts1 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		collide(&active_bodies, &contacts0, body_data, collider_data, body_connections, temporary);
		test_assert(contacts0.count == 1, "Incorrect number of contacts.");
		
		collider_data.boxes.transforms = box_transforms_reversed;
		
		collide(&active_bodies, &contacts1, body_data, collider_data, body_connections, temporary);
		test_assert(contacts1.count == 1, "Incorrect number of contacts.");
		
		for (unsigned i = 0; i < 1; ++i) {
			const float* p0 = contacts0.data[i].position;
			
			float distance = vector_distance_squared(contacts1.data[0].position, p0);
			unsigned index1 = 0;
			
			for (unsigned j = 0; j < 1; ++j) {
				test_assert(contacts0.bodies[i].a == contacts1.bodies[j].b, "Inconsistent bodies.");
				test_assert(contacts0.bodies[i].b == contacts1.bodies[j].a, "Inconsistent bodies.");
				
				const float* p1 = contacts1.data[j].position;
				float d = vector_distance_squared(p1, p0);
				
				if (d < distance) {
					distance = d;
					index1 = j;
				}
			}
			
			test_assert(distance < 1e-3f, "No match found for contact in swapped configuration.");
			test_assert((uint32_t)contacts0.tags[i] == (((uint32_t)contacts1.tags[index1] >> 16) | ((uint32_t)contacts1.tags[index1] << 16)), "Inconsistent tags between configurations.");
		}
	}
}

static void box_box_faces_share_tags(Arena arena) {
	// Tests ensuring that vertices and edges are shared between faces.
	printf("box_box_faces_share_tags\n");
	
	for (unsigned i = 0; i < (1 << 12); ++i) {
		Arena temporary = arena;
		
		static const unsigned box_count = 2;
		static const unsigned body_count = 2;
		static const unsigned max_contacts = box_count*64;
		
		int ax = (i >> 0) & 3;
		int ay = (i >> 2) & 3;
		int az = (i >> 4) & 3;
		
		int bx = (i >> 6) & 3;
		int by = (i >> 8) & 3;
		int bz = (i >> 10) & 3;
		
		SIMD_ALIGNED Transform body_transforms[body_count] = {
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		SIMD_ALIGNED uint16_t tags[box_count] = { 0, 1 };
		
		SIMD_ALIGNED BoxCollider boxes[box_count] = {
			{ { 1.125f, 1.125f, 1.125f } },
			{ { 1.125f, 1.125f, 1.125f } },
		};
		
		// Note that distance needs to be less than geometric because of edge-edge bias for feature stability.
		SIMD_ALIGNED Transform box_transforms0[box_count] = {
			{
				{ 2.0f, 0.1f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, 0.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms0[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms0[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms0[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms0[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		rotate_axis_angle(box_transforms0[0].rotation, 0.0f, 1.0f, 0.0f, pi*1e-3f);
		
		rotate_axis_angle(box_transforms0[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms0[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms0[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		
		SIMD_ALIGNED Transform box_transforms1[] = {
			{
				{ 2.0f, 0.1f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, 0.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms1[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms1[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms1[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms1[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		rotate_axis_angle(box_transforms1[0].rotation, 0.0f, 1.0f, 0.0f, -pi*1e-3f);
		
		rotate_axis_angle(box_transforms1[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms1[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms1[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		
		SIMD_ALIGNED BodyProperties properties[body_count] = {};
		SIMD_ALIGNED BodyMomentum momentum[body_count] = {};
		SIMD_ALIGNED uint8_t idle_counters[body_count] = {};
		
		BodyData body_data = {};
		
		body_data.transforms = body_transforms;
		body_data.properties = properties;
		body_data.momentum = momentum;
		body_data.idle_counters = idle_counters;
		body_data.count = body_count;
		
		ColliderData collider_data = {};
		
		collider_data.boxes.tags = tags;
		collider_data.boxes.data = boxes;
		collider_data.boxes.count = box_count;
		
		BodyConnections body_connections = {};
		
		SIMD_ALIGNED uint16_t body_indices[body_count] = {};
		ActiveBodies active_bodies = { body_indices, body_count };
		
		ContactData contacts0 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		ContactData contacts1 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		collider_data.boxes.transforms = box_transforms0;
		collide(&active_bodies, &contacts0, body_data, collider_data, body_connections, temporary);
		test_assert(contacts0.count == 2, "Incorrect number of contacts.");
		
		collider_data.boxes.transforms = box_transforms1;
		collide(&active_bodies, &contacts1, body_data, collider_data, body_connections, temporary);
		test_assert(contacts1.count == 2, "Incorrect number of contacts.");
		
		// Check that the edge intersection in the face contact contains the vertex.
		unsigned label_edge0 = (uint32_t)contacts0.tags[0];
		unsigned label_vertex0 = (uint32_t)contacts0.tags[1];
		
		if (contacts0.bodies[0].a != 0)
			label_edge0 = (label_edge0 >> 16) | (label_edge0 << 16);
		
		if (contacts0.bodies[1].a != 0)
			label_vertex0 = (label_vertex0 >> 16) | (label_vertex0 << 16);
		
		if ((label_vertex0 >> 16) != 0xffff) {
			unsigned t = label_vertex0;
			label_vertex0 = label_edge0;
			label_edge0 = t;
		}
		
		test_assert((label_vertex0 >> 16) == 0xffff, "Invalid vertex label.");
		test_assert((label_edge0 >> 16) != 0xffff, "Invalid edge label.");
		
		unsigned label_edge1 = (uint32_t)contacts1.tags[0];
		unsigned label_vertex1 = (uint32_t)contacts1.tags[1];
		
		if (contacts1.bodies[0].a != 0)
			label_edge1 = (label_edge1 >> 16) | (label_edge1 << 16);
		
		if (contacts1.bodies[1].a != 0)
			label_vertex1 = (label_vertex1 >> 16) | (label_vertex1 << 16);
		
		if ((label_vertex1 >> 16) != 0xffff) {
			unsigned t = label_vertex1;
			label_vertex1 = label_edge1;
			label_edge1 = t;
		}
		
		test_assert((label_vertex1 >> 16) == 0xffff, "Invalid vertex label.");
		test_assert((label_edge1 >> 16) != 0xffff, "Invalid edge label.");
		
		test_assert(label_vertex0 == label_vertex1, "Vertex not shared between faces.");
		test_assert(label_edge0 == label_edge1, "Edge not shared between faces.");
	}
}

static void box_box_consistent_face_edge_tags(Arena arena) {
	// Tests ensuring that edge contacts and face contacts have consistent edge tags.
	printf("box_box_consistent_face_edge_tags\n");
	
	for (unsigned i = 0; i < (1 << 12); ++i) {
		Arena temporary = arena;
		
		static const unsigned box_count = 2;
		static const unsigned body_count = 2;
		static const unsigned max_contacts = box_count*64;
		
		int ax = (i >> 0) & 3;
		int ay = (i >> 2) & 3;
		int az = (i >> 4) & 3;
		
		int bx = (i >> 6) & 3;
		int by = (i >> 8) & 3;
		int bz = (i >> 10) & 3;
		
		SIMD_ALIGNED Transform body_transforms[body_count] = {
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0, 0, 0 }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		SIMD_ALIGNED uint16_t tags[box_count] = { 0, 1 };
		
		SIMD_ALIGNED BoxCollider boxes[box_count] = {
			{ { 1.125f, 1.125f, 1.125f } },
			{ { 1.125f, 1.125f, 1.125f } },
		};
		
		// Note that distance needs to be less than geometric because of edge-edge bias for feature stability.
		SIMD_ALIGNED Transform box_transforms_edge[box_count] = {
			{
				{ 2.2f, 2.2f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, 0.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms_edge[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms_edge[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms_edge[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms_edge[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		rotate_axis_angle(box_transforms_edge[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.25f);
		
		rotate_axis_angle(box_transforms_edge[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms_edge[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms_edge[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		
		SIMD_ALIGNED Transform box_transforms_face[] = {
			{
				{ 2.0f, 0.1f, 0.0f }, 0,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
			{
				{ 0.0f, 0.0f, 0.0f }, 1,
				{ 0.0f, 0.0f, 0.0f, 1.0f },
			},
		};
		
		rotate_axis_angle(box_transforms_face[0].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*az);
		rotate_axis_angle(box_transforms_face[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*ay);
		rotate_axis_angle(box_transforms_face[0].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*ax);
		rotate_axis_angle(box_transforms_face[0].rotation, 0.0f, 1.0f, 0.0f, pi*0.25f);
		
		rotate_axis_angle(box_transforms_face[1].rotation, 0.0f, 0.0f, 1.0f, pi*0.5f*bz);
		rotate_axis_angle(box_transforms_face[1].rotation, 0.0f, 1.0f, 0.0f, pi*0.5f*by);
		rotate_axis_angle(box_transforms_face[1].rotation, 1.0f, 0.0f, 0.0f, pi*0.5f*bx);
		
		SIMD_ALIGNED BodyProperties properties[body_count] = {};
		SIMD_ALIGNED BodyMomentum momentum[body_count] = {};
		SIMD_ALIGNED uint8_t idle_counters[body_count] = {};
		
		BodyData body_data = {};
		
		body_data.transforms = body_transforms;
		body_data.properties = properties;
		body_data.momentum = momentum;
		body_data.idle_counters = idle_counters;
		body_data.count = body_count;
		
		ColliderData collider_data = {};
		
		collider_data.boxes.tags = tags;
		collider_data.boxes.data = boxes;
		collider_data.boxes.count = box_count;
		
		BodyConnections body_connections = {};
		
		SIMD_ALIGNED uint16_t body_indices[body_count] = {};
		ActiveBodies active_bodies = { body_indices, body_count };
		
		ContactData contacts0 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		ContactData contacts1 = {
			allocate_array<Contact>(&temporary, max_contacts, 64),
			allocate_array<BodyPair>(&temporary, max_contacts, 64),
			allocate_array<uint64_t>(&temporary, max_contacts, 64),
			max_contacts,
		};
		
		collider_data.boxes.transforms = box_transforms_edge;
		collide(&active_bodies, &contacts0, body_data, collider_data, body_connections, temporary);
		test_assert(contacts0.count == 1, "Incorrect number of contacts.");
		
		collider_data.boxes.transforms = box_transforms_face;
		collide(&active_bodies, &contacts1, body_data, collider_data, body_connections, temporary);
		test_assert(contacts1.count == 2, "Incorrect number of contacts.");
		
		// Check that the edge intersection in the face contact contains the vertex.
		unsigned label_edge = (uint32_t)contacts1.tags[0];
		unsigned label_vertex = (uint32_t)contacts1.tags[1];
		
		if (contacts1.bodies[0].a != 0)
			label_edge = (label_edge >> 16) | (label_edge << 16);
		
		if (contacts1.bodies[1].a != 0)
			label_vertex = (label_vertex >> 16) | (label_vertex << 16);
		
		if ((label_vertex >> 16) != 0xffff) {
			unsigned t = label_vertex;
			label_vertex = label_edge;
			label_edge = t;
		}
		
		test_assert((label_vertex >> 16) == 0xffff, "Invalid vertex label.");
		test_assert((label_edge >> 16) != 0xffff, "Invalid edge label.");
		
		label_vertex &= 0xff;
		
		test_assert(label_vertex == ((label_edge >> 0) & 0xff) ||
					label_vertex == ((label_edge >> 8) & 0xff) ||
					label_vertex == ((label_edge >> 16) & 0xff) ||
					label_vertex == ((label_edge >> 24) & 0xff), "Edge does not contain the vertex.");
		
		// Check that the edge intersection in the first case is present in one of the face contacts (there's one vertex contact and one edge intersection).
		unsigned matches = 0;
		
		for (unsigned i = 0; i < 2; ++i) {
			if (contacts0.bodies[0].a == contacts1.bodies[i].a && contacts0.bodies[0].b == contacts1.bodies[i].b) {
				if (contacts0.tags[0] == contacts1.tags[i])
					++matches;
			}
			else if (contacts0.bodies[0].b == contacts1.bodies[i].a && contacts0.bodies[0].a == contacts1.bodies[i].b) {
				if (((contacts0.tags[0] << 16) | (contacts0.tags[0] >> 16)) == contacts1.tags[i])
					++matches;
			}
		}
		
		test_assert(matches == 1, "Incorrect number of matches.");
	}
}

int main() {
	Arena arena = {};
	arena.size = 64*1024*1024;
	arena.data = _mm_malloc(arena.size, 4096);
	
	box_box_test_case_0(arena);
	box_box_face_face_tags_0(arena);
	box_box_face_face_tags_1(arena);
	box_box_edge_edge_tags(arena);
	box_box_faces_share_tags(arena);
	box_box_consistent_face_edge_tags(arena);
	
	printf("All tests passed.\n");
	
	return 0;
}
