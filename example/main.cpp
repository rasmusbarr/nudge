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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/GLUT.h>
#include <OpenGL/gl.h>
#else
#include <GLUT/glut.h>
#include <gl/gl.h>
#endif

static const unsigned max_body_count = 2048;
static const unsigned max_box_count = 2048;
static const unsigned max_sphere_count = 2048;

static const nudge::Transform identity_transform = { {}, 0, { 0.0f, 0.0f, 0.0f, 1.0f } };

static nudge::Arena arena;
static nudge::BodyData bodies;
static nudge::ColliderData colliders;
static nudge::ContactData contact_data;
static nudge::ContactCache contact_cache;
static nudge::ActiveBodies active_bodies;

static inline void quaternion_concat(float r[4], const float a[4], const float b[4]) {
	r[0] = b[0]*a[3] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1];
	r[1] = b[1]*a[3] + a[1]*b[3] + a[2]*b[0] - a[0]*b[2];
	r[2] = b[2]*a[3] + a[2]*b[3] + a[0]*b[1] - a[1]*b[0];
	r[3] = a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2];
}

static inline void quaternion_transform(float r[3], const float a[4], const float b[3]) {
	float t[3];
	t[0] = a[1]*b[2] - a[2]*b[1];
	t[1] = a[2]*b[0] - a[0]*b[2];
	t[2] = a[0]*b[1] - a[1]*b[0];
	
	t[0] += t[0];
	t[1] += t[1];
	t[2] += t[2];
	
	r[0] = b[0] + a[3]*t[0] + a[1]*t[2] - a[2]*t[1];
	r[1] = b[1] + a[3]*t[1] + a[2]*t[0] - a[0]*t[2];
	r[2] = b[2] + a[3]*t[2] + a[0]*t[1] - a[1]*t[0];
}

static inline void matrix(float r[16], const float s[3], const float q[4], const float t[3]) {
	float kx = q[0] + q[0];
	float ky = q[1] + q[1];
	float kz = q[2] + q[2];
	
	float xx = kx*q[0];
	float yy = ky*q[1];
	float zz = kz*q[2];
	float xy = kx*q[1];
	float xz = kx*q[2];
	float yz = ky*q[2];
	float sx = kx*q[3];
	float sy = ky*q[3];
	float sz = kz*q[3];
	
	r[0] = (1.0f - yy - zz) * s[0];
	r[1] = (xy + sz) * s[0];
	r[2] = (xz - sy) * s[0];
	r[3] = 0.0f;
	
	r[4] = (xy - sz) * s[1];
	r[5] = (1.0f - xx - zz) * s[1];
	r[6] = (yz + sx) * s[1];
	r[7] = 0.0f;
	
	r[8] = (xz + sy) * s[2];
	r[9] = (yz - sx) * s[2];
	r[10] = (1.0f - xx - yy) * s[2];
	r[11] = 0.0f;
	
	r[12] = t[0];
	r[13] = t[1];
	r[14] = t[2];
	r[15] = 1.0f;
}

static inline unsigned add_box(float mass, float cx, float cy, float cz) {
	if (bodies.count == max_body_count || colliders.boxes.count == max_box_count)
		return 0;
	
	unsigned body = bodies.count++;
	unsigned collider = colliders.boxes.count++;
	
	float k = mass * (1.0f/3.0f);
	
	float kcx2 = k*cx*cx;
	float kcy2 = k*cy*cy;
	float kcz2 = k*cz*cz;
	
	nudge::BodyProperties properties = {};
	properties.mass_inverse = 1.0f / mass;
	properties.inertia_inverse[0] = 1.0f / (kcy2+kcz2);
	properties.inertia_inverse[1] = 1.0f / (kcx2+kcz2);
	properties.inertia_inverse[2] = 1.0f / (kcx2+kcy2);
	
	memset(&bodies.momentum[body], 0, sizeof(bodies.momentum[body]));
	bodies.idle_counters[body] = 0;
	bodies.properties[body] = properties;
	bodies.transforms[body] = identity_transform;
	
	colliders.boxes.transforms[collider] = identity_transform;
	colliders.boxes.transforms[collider].body = body;
	
	colliders.boxes.data[collider].size[0] = cx;
	colliders.boxes.data[collider].size[1] = cy;
	colliders.boxes.data[collider].size[2] = cz;
	colliders.boxes.tags[collider] = collider;
	
	return body;
}

static inline unsigned add_sphere(float mass, float radius) {
	if (bodies.count == max_body_count || colliders.spheres.count == max_sphere_count)
		return 0;
	
	unsigned body = bodies.count++;
	unsigned collider = colliders.spheres.count++;
	
	float k = 2.5f / (mass*radius*radius);
	
	nudge::BodyProperties properties = {};
	properties.mass_inverse = 1.0f / mass;
	properties.inertia_inverse[0] = k;
	properties.inertia_inverse[1] = k;
	properties.inertia_inverse[2] = k;
	
	memset(&bodies.momentum[body], 0, sizeof(bodies.momentum[body]));
	bodies.idle_counters[body] = 0;
	bodies.properties[body] = properties;
	bodies.transforms[body] = identity_transform;
	
	colliders.spheres.transforms[collider] = identity_transform;
	colliders.spheres.transforms[collider].body = body;
	
	colliders.spheres.data[collider].radius = radius;
	colliders.spheres.tags[collider] = collider + max_box_count;
	
	return body;
}

static void render() {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// Setup projection.
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	glMatrixMode(GL_PROJECTION);
	
	float fov = 0.25f;
	float aspect = (float)viewport[2]/(float)viewport[3];
	float near_z = 1.0f;
	float far_z = 1000.0f;
	{
		float dy = 2.0f * near_z * tanf(fov);
		float zn = 2.0f * near_z;
		float dz = far_z - near_z;
		
		float m[16] = {
			zn / (dy * aspect), 0.0f, 0.0f, 0.0f,
			0.0f, zn / dy, 0.0f, 0.0f,
			0.0f, 0.0f, (-far_z - near_z) / dz, -1.0f,
			0.0f, 0.0f, (-zn * far_z) / dz, 0.0f,
		};
		
		glLoadMatrixf(m);
	}
	
	glTranslatef(0.0f, 0.0f, -75.0f);
	
	// Switch to model matrix.
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	// Setup light.
	GLfloat light_ambient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat light_direction[] = { 1.0f, 1.0f, 1.0f, 0.0f };
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, light_direction);
	
	// Render boxes.
	for (unsigned i = 0; i < colliders.boxes.count; ++i) {
		unsigned body = colliders.boxes.transforms[i].body;
		
		float scale[3];
		float rotation[4];
		float position[3];
		
		memcpy(scale, colliders.boxes.data[i].size, sizeof(scale));
		
		quaternion_concat(rotation, bodies.transforms[body].rotation, colliders.boxes.transforms[i].rotation);
		quaternion_transform(position, bodies.transforms[body].rotation, colliders.boxes.transforms[i].position);
		
		position[0] += bodies.transforms[body].position[0];
		position[1] += bodies.transforms[body].position[1];
		position[2] += bodies.transforms[body].position[2];
		
		float m[16];
		matrix(m, scale, rotation, position);
		
		glLoadMatrixf(m);
		glutSolidCube(2.0f);
	}
	
	// Render spheres.
	for (unsigned i = 0; i < colliders.spheres.count; ++i) {
		unsigned body = colliders.spheres.transforms[i].body;
		
		float scale[3];
		float rotation[4];
		float position[3];
		
		scale[0] = scale[1] = scale[2] = colliders.spheres.data[i].radius;
		
		quaternion_concat(rotation, bodies.transforms[body].rotation, colliders.spheres.transforms[i].rotation);
		quaternion_transform(position, bodies.transforms[body].rotation, colliders.spheres.transforms[i].position);
		
		position[0] += bodies.transforms[body].position[0];
		position[1] += bodies.transforms[body].position[1];
		position[2] += bodies.transforms[body].position[2];
		
		float m[16];
		matrix(m, scale, rotation, position);
		
		glLoadMatrixf(m);
		glutSolidSphere(1.0f, 16, 8);
	}
	
	glutSwapBuffers();
}

static void simulate() {
	static const unsigned steps = 2;
	static const unsigned iterations = 20;
	
	float time_step = 1.0f / (60.0f * (float)steps);
	
	for (unsigned n = 0; n < steps; ++n) {
		// Setup a temporary memory arena. The same temporary memory is reused each iteration.
		nudge::Arena temporary = arena;
		
		// Find contacts.
		nudge::BodyConnections connections = {}; // NOTE: Custom constraints should be added as body connections.
		nudge::collide(&active_bodies, &contact_data, bodies, colliders, connections, temporary);
		
		// NOTE: Custom contacts can be added here, e.g., against the static environment.
		
		// Apply gravity and damping.
		float damping = 1.0f - time_step*0.25f;
		
		for (unsigned i = 0; i < active_bodies.count; ++i) {
			unsigned index = active_bodies.indices[i];
			
			bodies.momentum[index].velocity[1] -= 9.82f * time_step;
			
			bodies.momentum[index].velocity[0] *= damping;
			bodies.momentum[index].velocity[1] *= damping;
			bodies.momentum[index].velocity[2] *= damping;
			
			bodies.momentum[index].angular_velocity[0] *= damping;
			bodies.momentum[index].angular_velocity[1] *= damping;
			bodies.momentum[index].angular_velocity[2] *= damping;
		}
		
		// Read previous impulses from contact cache.
		nudge::ContactImpulseData* contact_impulses = nudge::read_cached_impulses(contact_cache, contact_data, &temporary);
		
		// Setup contact constraints and apply the initial impulses.
		nudge::ContactConstraintData* contact_constraints = nudge::setup_contact_constraints(active_bodies, contact_data, bodies, contact_impulses, &temporary);
		
		// Apply contact impulses. Increasing the number of iterations will improve stability.
		for (unsigned i = 0; i < iterations; ++i) {
			nudge::apply_impulses(contact_constraints, bodies);
			// NOTE: Custom constraint impulses should be applied here.
		}
		
		// Update contact impulses.
		nudge::update_cached_impulses(contact_constraints, contact_impulses);
		
		// Write the updated contact impulses to the cache.
		nudge::write_cached_impulses(&contact_cache, contact_data, contact_impulses);
		
		// Move active bodies.
		nudge::advance(active_bodies, bodies, time_step);
	}
}

static void timer(int) {
	glutPostRedisplay();
	glutTimerFunc(16, timer, 0);
	simulate();
}

int main(int argc, const char* argv[]) {
	// Disable denormals for performance.
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	
	// Print information about instruction set.
#ifdef __AVX2__
	printf("Using 8-wide AVX\n");
#else
	printf("Using 4-wide SSE\n");
#if defined(__SSE4_1__) || defined(__AVX__)
	printf("BLENDVPS: Enabled\n");
#else
	printf("BLENDVPS: Disabled\n");
#endif
#endif
	
#ifdef __FMA__
	printf("FMA: Enabled\n");
#else
	printf("FMA: Disabled\n");
#endif
	
	// Allocate memory for simulation arena.
	arena.size = 64*1024*1024;
	arena.data = _mm_malloc(arena.size, 4096);
	
	// Allocate memory for bodies, colliders, and contacts.
	active_bodies.capacity = max_box_count;
	active_bodies.indices = static_cast<uint16_t*>(_mm_malloc(sizeof(uint16_t)*max_body_count, 64));
	
	bodies.idle_counters = static_cast<uint8_t*>(_mm_malloc(sizeof(uint8_t)*max_body_count, 64));
	bodies.transforms = static_cast<nudge::Transform*>(_mm_malloc(sizeof(nudge::Transform)*max_body_count, 64));
	bodies.momentum = static_cast<nudge::BodyMomentum*>(_mm_malloc(sizeof(nudge::BodyMomentum)*max_body_count, 64));
	bodies.properties = static_cast<nudge::BodyProperties*>(_mm_malloc(sizeof(nudge::BodyProperties)*max_body_count, 64));
	
	colliders.boxes.data = static_cast<nudge::BoxCollider*>(_mm_malloc(sizeof(nudge::BoxCollider)*max_box_count, 64));
	colliders.boxes.tags = static_cast<uint16_t*>(_mm_malloc(sizeof(uint16_t)*max_box_count, 64));
	colliders.boxes.transforms = static_cast<nudge::Transform*>(_mm_malloc(sizeof(nudge::Transform)*max_box_count, 64));
	
	colliders.spheres.data = static_cast<nudge::SphereCollider*>(_mm_malloc(sizeof(nudge::SphereCollider)*max_sphere_count, 64));
	colliders.spheres.tags = static_cast<uint16_t*>(_mm_malloc(sizeof(uint16_t)*max_sphere_count, 64));
	colliders.spheres.transforms = static_cast<nudge::Transform*>(_mm_malloc(sizeof(nudge::Transform)*max_sphere_count, 64));
	
	contact_data.capacity = max_body_count*64;
	contact_data.bodies = static_cast<nudge::BodyPair*>(_mm_malloc(sizeof(nudge::BodyPair)*contact_data.capacity, 64));
	contact_data.data = static_cast<nudge::Contact*>(_mm_malloc(sizeof(nudge::Contact)*contact_data.capacity, 64));
	contact_data.tags = static_cast<uint64_t*>(_mm_malloc(sizeof(uint64_t)*contact_data.capacity, 64));
	contact_data.sleeping_pairs = static_cast<uint32_t*>(_mm_malloc(sizeof(uint32_t)*contact_data.capacity, 64));
	
	contact_cache.capacity = max_body_count*64;
	contact_cache.data = static_cast<nudge::CachedContactImpulse*>(_mm_malloc(sizeof(nudge::CachedContactImpulse)*contact_cache.capacity, 64));
	contact_cache.tags = static_cast<uint64_t*>(_mm_malloc(sizeof(uint64_t)*contact_cache.capacity, 64));
	
	// The first body is the static world.
	bodies.count = 1;
	bodies.idle_counters[0] = 0;
	bodies.transforms[0] = identity_transform;
	memset(bodies.momentum, 0, sizeof(bodies.momentum[0]));
	memset(bodies.properties, 0, sizeof(bodies.properties[0]));
	
	// Add ground.
	{
		unsigned collider = colliders.boxes.count++;
		
		colliders.boxes.transforms[collider] = identity_transform;
		colliders.boxes.transforms[collider].position[1] -= 20.0f;
		
		colliders.boxes.data[collider].size[0] = 400.0f;
		colliders.boxes.data[collider].size[1] = 10.0f;
		colliders.boxes.data[collider].size[2] = 400.0f;
		colliders.boxes.tags[collider] = collider;
	}
	
	// Add boxes.
	for (unsigned i = 0; i < 1024; ++i) {
		float sx = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		float sy = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		float sz = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		
		unsigned body = add_box(8.0f*sx*sy*sz, sx, sy, sz);
		
		bodies.transforms[body].position[0] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
		bodies.transforms[body].position[1] += (float)rand() * (1.0f/(float)RAND_MAX) * 300.0f;
		bodies.transforms[body].position[2] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
	}
	
	// Add spheres.
	for (unsigned i = 0; i < 512; ++i) {
		float s = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		
		unsigned body = add_sphere(4.18879f*s*s*s, s);
		
		bodies.transforms[body].position[0] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
		bodies.transforms[body].position[1] += (float)rand() * (1.0f/(float)RAND_MAX) * 300.0f;
		bodies.transforms[body].position[2] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
	}
	
	// Start GLUT.
	glutInit(&argc, const_cast<char**>(argv));
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(1024, 600);
	glutCreateWindow("nudge");
	glutDisplayFunc(render);
	
	timer(0);
	
	glutMainLoop();
	
	return 0;
}
