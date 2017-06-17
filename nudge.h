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

#ifndef NUDGE_H
#define NUDGE_H

#include <stdint.h>

namespace nudge {
	struct Arena {
		void* data;
		uintptr_t size;
	};
	
	struct Transform {
		float position[3];
		uint32_t body;
		float rotation[4];
	};
	
	struct BodyProperties {
		float inertia_inverse[3];
		float mass_inverse;
	};
	
	struct BodyMomentum {
		float velocity[3];
		float unused0;
		float angular_velocity[3];
		float unused1;
	};
	
	struct SphereCollider {
		float radius;
	};
	
	struct BoxCollider {
		float size[3];
		float unused;
	};
	
	struct Contact {
		float position[3];
		float penetration;
		float normal[3];
		float friction;
	};
	
	struct BodyPair {
		uint16_t a;
		uint16_t b;
	};
	
	struct ContactData {
		Contact* data;
		BodyPair* bodies;
		uint64_t* tags;
		uint32_t capacity;
		uint32_t count;
		
		uint32_t* sleeping_pairs;
		uint32_t sleeping_count;
	};
	
	struct ColliderData {
		struct {
			uint16_t* tags;
			BoxCollider* data;
			Transform* transforms;
			uint32_t count;
		} boxes;
		
		struct {
			uint16_t* tags;
			SphereCollider* data;
			Transform* transforms;
			uint32_t count;
		} spheres;
	};
	
	struct BodyData {
		Transform* transforms;
		BodyProperties* properties;
		BodyMomentum* momentum;
		uint8_t* idle_counters;
		uint32_t count;
	};
	
	struct BodyConnections {
		BodyPair* data;
		uint32_t count;
	};
	
	struct CachedContactImpulse {
		float impulse[3];
		float unused;
	};
	
	struct ContactCache {
		uint64_t* tags;
		CachedContactImpulse* data;
		uint32_t capacity;
		uint32_t count;
	};
	
	struct ActiveBodies {
		uint16_t* indices;
		uint32_t capacity;
		uint32_t count;
	};
	
	struct ContactImpulseData;
	struct ContactConstraintData;
	
	void collide(ActiveBodies* active_bodies, ContactData* contacts, BodyData bodies, ColliderData colliders, BodyConnections body_connections, Arena temporary);
	
	ContactImpulseData* read_cached_impulses(ContactCache contact_cache, ContactData contacts, Arena* memory);
	
	void write_cached_impulses(ContactCache* contact_cache, ContactData contacts, ContactImpulseData* contact_impulses);
	
	ContactConstraintData* setup_contact_constraints(ActiveBodies active_bodies, ContactData contacts, BodyData bodies, ContactImpulseData* contact_impulses, Arena* memory);
	
	void apply_impulses(ContactConstraintData* data, BodyData bodies);
	
	void update_cached_impulses(ContactConstraintData* data, ContactImpulseData* contact_impulses);
	
	void advance(ActiveBodies active_bodies, BodyData bodies, float time_step);
}

#endif
