//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2018-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Ponekker Patrik
// Neptun : VDRDYH
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#define _USE_MATH_DEFINES		// M_PI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int screenWidth = 600, screenHeight = 600;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

const float EPSILON = 0.0001f;
const int MAXRECURSION = 10;

struct vec3 {
	float x, y, z;
	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { 
		x = x0; y = y0; z = z0;
	}

	vec3(const vec3& v) {
		x = v.x;
		y = v.y;
		z = v.z;
	}

	vec3& operator=(const vec3& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	vec3 operator*(float a) const { return vec3(
		x * a, y * a, z * a);
	}
	vec3 operator/(float d) const {
		return vec3(x / d, y / d, z / d); 
	}
	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	void operator+=(const vec3& v) {
		x += v.x; y += v.y; z += v.z;
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z); 
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z); 
	}
	vec3 operator-() const { 
		return vec3(-x, -y, -z);
	}
	vec3 normalize() { 
		return *this = *this * (1.0f / sqrtf(x * x + y * y + z * z));
	}
	float Length() const { 
		return sqrtf(x * x + y * y + z * z);
	}
	operator float*() {
		return &x; 
	}

	float dot(const vec3& v) const {
		return x * v.x + y * v.y + z * v.z;
	}

};

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

class Ray{
public: 
	vec3 P0; // kezdopont
	vec3 dv; // iranyvektor
};

class Light {
public:
	vec3 color;	// szine
	vec3 P0;	// helyzete
};

class Camera{
public:
	vec3 P0; // kamera helyzete
};

class Object {
public:
	vec3 color;		// szin
	vec3 Kr;			// Fresnel-egyutthato (szamolando)
	vec3 fr;			// toresi tenyezo
	vec3 kappa;		// kioltasi tenyezo
	bool isReflective;  // tukrozo felulet
	bool isRefractive;  // toro felulet

	virtual float intersect(Ray& ray) = 0;
	virtual vec3 getNormal(vec3 intersect) = 0; // feluleti normalist adott pontban lekerdezo fuggveny

	void computeFresnel(float costheta) {
		Kr.x = ((pow((fr.x - 1.0), 2)) + (pow(kappa.x, 2)) + (pow((1.0 - costheta), 5)) * (4 * fr.x)) /
			((pow((fr.x + 1.0), 2)) + (pow(kappa.x, 2)));
		Kr.y = ((pow((fr.y - 1.0), 2)) + (pow(kappa.y, 2)) + (pow((1.0 - costheta), 5)) * (4 * fr.y)) /
			((pow((fr.y + 1.0), 2)) + (pow(kappa.y, 2)));
		Kr.z = ((pow((fr.z - 1.0), 2)) + (pow(kappa.z, 2)) + (pow((1.0 - costheta), 5)) * (4 * fr.z)) /
			((pow((fr.z + 1.0), 2)) + (pow(kappa.z, 2)));

	}
};

class Scene {
public:
	Object* objects[100];
	int objectCount;
	Light light;

	Scene() {
		light.color = vec3(0.8f, 0.8f, 0.8f);
		light.P0 = vec3(200.0f, 200.0f, 0.0f);
		objectCount = 0;
	}

	void add(Object* object) {
		objects[objectCount] = object;
		objectCount++;
	}

	void render(vec3 image[]) {
		float w2 = screenWidth / 2.0f;
		float h2 = screenHeight / 2.0f;
		Camera cam;

		float FOV = 60.0f;
		float FOV2 = FOV / 2.0f;
		float L = w2 / tanf(FOV2 * M_PI / 180.0f);

		cam.P0.x = 0.0f;
		cam.P0.y = 0.0f;
		cam.P0.z = -1.0f * L;

		for (int y = 0; y < screenHeight; y++) {
			for (int x = 0; x < screenWidth; x++) {
				Ray ray;
				ray.P0 = vec3(x - w2, y - h2, 0.0f);
				ray.dv = (ray.P0 - cam.P0);
				ray.dv.normalize();
				image[y * screenWidth + x] = Trace(ray, 0);
			}
		}
	}

	vec3 Trace(Ray& ray, int iterat) {
		vec3 color;
		color.x = color.y = color.z = 0.0f;
		if (iterat < MAXRECURSION) {
			iterat++;
			float t = -1.0f;
			float mint = 9999999999999999.9999999999999999f;
			int minindex = -1;
			for (int i = 0; i < objectCount; i++) {
				t = objects[i]->intersect(ray);
				if (t > EPSILON) {
					if (t < mint) {
						mint = t;
						minindex = i;
					}
				}
			}
			if (minindex >= 0) {
				t = mint;
				vec3 intersectPoint;
				intersectPoint = ((ray.P0) + (ray.dv * t));
				vec3 normal = objects[minindex]->getNormal(intersectPoint);
				Ray iRay; // sugar ami a metszespontbol a feny fele fog majd nezni
				iRay.P0 = intersectPoint + normal * EPSILON;
				iRay.dv = light.P0 - intersectPoint;
				iRay.dv.normalize();
				float factor = normal.dot(iRay.dv); // ez a cos(theta)
				if (factor < 0.0f) {
					factor = 0.0f;
				}
				color = objects[minindex]->color;
				if (!objects[minindex]->isReflective && !objects[minindex]->isRefractive) {
					color.x = color.x * (light.color.x) * factor;
					color.y = color.y * (light.color.y) * factor;
					color.z = color.z * (light.color.z) * factor;
				}
				if (objects[minindex]->isReflective) {
					objects[minindex]->computeFresnel(factor);
					float costheta2 = -1.0f * ray.dv.dot(normal);

					Ray rRay;
					rRay.P0 = intersectPoint + normal * EPSILON;
					rRay.dv = ray.dv + normal * 2.0f * costheta2;
					rRay.dv.normalize();
					vec3 plusColor = Trace(rRay, iterat);

					color = color + plusColor;

					color.x = color.x * objects[minindex]->Kr.x;
					color.y = color.y * objects[minindex]->Kr.y;
					color.z = color.z * objects[minindex]->Kr.z;
				}

				if (objects[minindex]->isRefractive) {
					float n = objects[minindex]->fr.x; // mert a felulet mindne hullamhosszon ugyanugy tor, amugy ki kell majd szamolni a tobbire
					vec3 tnormal = normal;
					float cosalpha = -1.0f * ray.dv.dot(tnormal);
					if (cosalpha < 0.0f) {
						n = 1.0f / n;
						tnormal = tnormal * -1.0f;
						cosalpha = -1.0f * ray.dv.dot(tnormal);
					}
					float disc = 1.0f - ((1.0f - cosalpha * cosalpha) / (n * n));
					if (disc > 0.0f) {
						Ray fRay;
						fRay.P0 = intersectPoint + tnormal * EPSILON * -1.0f;
						fRay.dv = ray.dv / n + tnormal * (cosalpha / n - sqrt(disc));
						fRay.dv.normalize();
						vec3 plusColor = Trace(fRay, iterat);
						color = plusColor;
					}
				}
			}
		}
		return color;
	}
};

class Sphere : public Object {
public: 
	vec3 origo;
	float radius;

	Sphere(vec3 o = vec3(0.0f, 0.0f, 0.0f), float r = 1.0f) {
		origo = o;
		radius = r;
	}

	float intersect(Ray& ray) {
		float dx = ray.dv.x;
		float dy = ray.dv.y;
		float dz = ray.dv.z;
		float x0 = ray.P0.x;
		float y0 = ray.P0.y;
		float z0 = ray.P0.z;
		float cx = origo.x;
		float cy = origo.y;
		float cz = origo.z;
		float R = radius;

		float a = dx * dx + dy * dy + dz * dz;
		float b = 2.0f * dx * (x0 - cx) + 2.0f * dy * (y0 - cy) + 2.0f * dz * (z0 - cz);
		float c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0 - 2.0f * (cx * x0 + cy * y0 + cz * z0) - R * R;

		float d = b * b - 4.0f * a * c;

		if (d < 0.0f) {
			return -1.0f;
		}
		float t = (-1.0f * b - sqrtf(d)) / (2.0f * a);

		if (t > EPSILON) {
			return t;
		}
		else {
			return 0.0f;
		}
	}

	vec3 getNormal(vec3 intersect) {
		return (intersect - origo).normalize();
	}
};

class Plane : public Object {
public:
	vec3 point;
	vec3 normal;

	Plane(vec3 p, vec3 n) {
		point = p;
		normal = n;
	}

	float intersect(Ray& ray) {
		float d = normal.dot(ray.dv);
			if (d == 0.0f) {
				return -1.0f;
			}
			float nx = normal.x;
			float ny = normal.y;
			float nz = normal.z;
			float Psx = point.x;
			float Psy = point.y;
			float Psz = point.z;

			float dvx = ray.dv.x;
			float dvy = ray.dv.y;
			float dvz = ray.dv.z;
			float Pex = ray.P0.x;
			float Pey = ray.P0.y;
			float Pez = ray.P0.z;

			float t = -1.0f * ((nx * Pex - nx * Psx + ny * Pey - ny * Psy + nz * Pez - nz * Psz) / (nx * dvx + ny * dvy + nz * dvz));

			if (t > EPSILON) return t;
			if (t > 0.0f) return 0.0f;

			return -1.0f;	
	}

	vec3 getNormal(vec3) {
		return normal;
	}
};

class Triangle : public Object {
public:
	vec3 A, B, C;

	Triangle(vec3 a, vec3 b, vec3 c) {
		A = a;
		B = b;
		C = c;
	}

	float intersect(Ray& ray) {
		vec3 normal = cross(B - A, C - A);

		//float t = ((A - ray.P0).dot(normal)) / (ray.dv.dot(normal));

		if (fabs(ray.dv.dot(normal)) < EPSILON) {
			return -1.0f;
		}

		float nx = normal.x;
		float ny = normal.y;
		float nz = normal.z;
		float Psx = A.x;
		float Psy = A.y;
		float Psz = A.z;

		float dvx = ray.dv.x;
		float dvy = ray.dv.y;
		float dvz = ray.dv.z;
		float Pex = ray.P0.x;
		float Pey = ray.P0.y;
		float Pez = ray.P0.z;

		float t = -1.0f * ((nx * Pex - nx * Psx + ny * Pey - ny * Psy + nz * Pez - nz * Psz) / (nx * dvx + ny * dvy + nz * dvz));

		if (t < 0.0f) {
			return -1.0f;
		}
		vec3 p = ray.P0 + (ray.dv * t);

		bool first = (cross(B - A, p - A)).dot(normal) > 0;
		if (!first) { return -1.0f; }
		bool scond = (cross(C - B, p - B)).dot(normal) > 0;
		if (!scond) { return -1.0f; }
		bool third = (cross(A - C, p - C)).dot(normal) > 0;
		if (!third) { return -1.0f; }
			
		if (first && scond && third) {
			if (t > EPSILON) return t;
			if (t > 0.0f) return 0.0f;
		}
		return -1.0f;
	}

	vec3 getNormal(vec3) {
		return cross(B - A, C - A);
	}
};






void getErrorInfo(unsigned int handle) {
	int logLen, written;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec3 image[]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

																	  // Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenWidth, screenHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;
vec3 image[screenWidth * screenHeight];	// The image, which stores the ray tracing result
Scene scene;
										// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, screenWidth, screenHeight);

	int i = 0;
	//zold gomb
	scene.add(new Sphere(vec3(0.0f, -200.0f, 300.0f), 100.0f));
	scene.objects[i]->color = vec3(0.1f, 0.8f, 0.1f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = false;

	//kek gomb
	scene.add(new Sphere(vec3(150.0f, -200.0f, 500.0f), 100.0f));
	scene.objects[i]->color = vec3(0.1f, 0.1f, 0.8f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = false;

	//floor
	scene.add(new Plane(vec3(0.0f, -300.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f)));
	scene.objects[i]->color = vec3(1.0f, 1.0f, 1.0f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = false;

	//left wall
	scene.add(new Plane(vec3(-300.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)));
	scene.objects[i]->color = vec3(0.1f, 0.1f, 0.9f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = false;

	//right wall
	scene.add(new Plane(vec3(300.0f, 0.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f)));
	scene.objects[i]->color = vec3(0.1f, 0.1f, 0.9f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = false;

	//cieling
	scene.add(new Plane(vec3(0.0f, 300.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)));
	scene.objects[i]->color = vec3(0.1f, 0.1f, 0.1f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = true;
	scene.objects[i++]->isRefractive = false;

	//front wall
	scene.add(new Plane(vec3(0.0f, 0.0f, 600.0f), vec3(0.0f, 0.0f, -1.0f)));
	scene.objects[i]->color = vec3(0.1f, 0.1f, 0.1f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = true;
	scene.objects[i++]->isRefractive = false;

	//back wall
	scene.add(new Plane(vec3(0.0f, 0.0f, -600.0f), vec3(0.0f, 0.0f, 1.0f)));
	scene.objects[i]->color = vec3(0.1f, 0.1f, 0.1f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = false;


	//uveggomb
	/*scene.add(new Sphere(vec3(35.0f, -200.0f, 150.0f), 100));
	scene.objects[i]->color = vec3(0.0f, 0.0f, 0.0f);
	scene.objects[i]->fr = vec3(1.13f, 1.13f, 1.13f);
	scene.objects[i]->kappa = vec3(1.0f, 1.0f, 1.0f);
	scene.objects[i]->isReflective = true;
	scene.objects[i++]->isRefractive = true;*/

	//arany gomb
	scene.add(new Sphere(vec3(-150.0f, -200.0f, 150.0f), 100));
	scene.objects[i]->color = vec3(218.0f / 255.0f, 165.0f / 255.0f, 32.0f / 255.0f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = true;
	scene.objects[i++]->isRefractive = false;

	vec3 A = vec3(0.0f, 0.0f, 100.0f);
	vec3 B = vec3(0.0f, 50.0f, 200.0f);
	vec3 C = vec3(50.0f, -50.0f, 200.0f);
	vec3 D = vec3(-50.0f, -50.0f, 200.0f);
	vec3 E = vec3(-50.0f, -50.0f, 200.0f);
	vec3 F = vec3(-50.0f, -50.0f, 200.0f);
	vec3 G = vec3(-50.0f, -50.0f, 200.0f);
	vec3 H = vec3(-50.0f, -50.0f, 200.0f);

	//1triangle
	scene.add(new Triangle(A, C, B));
	scene.objects[i]->color = vec3(0.1f, 0.9f, 0.1f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = true;
	scene.objects[i++]->isRefractive = true;

	//2triangle
	scene.add(new Triangle(B, A, D));
	scene.objects[i]->color = vec3(0.1f, 0.1f, 0.9f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = true;
	scene.objects[i++]->isRefractive = true;

	//3triangle
	scene.add(new Triangle(B, D, C));
	scene.objects[i]->color = vec3(0.9f, 0.1f, 0.1f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = true;
	scene.objects[i++]->isRefractive = true;
	
	//4triangle
	scene.add(new Triangle(A, C, D));
	scene.objects[i]->color = vec3(0.1f, 0.4f, 0.6f);
	scene.objects[i]->fr = vec3(0.17f, 0.35f, 1.5f);
	scene.objects[i]->kappa = vec3(3.1f, 2.7f, 1.9f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = true;

	//acquarium wall
	/*scene.add(new Plane(vec3(0.0f, 0.0f, 50.0f), vec3(0.0f, 0.0f, 1.0f)));
	scene.objects[i]->color = vec3(0.0f, 0.0f, 0.0f);
	scene.objects[i]->fr = vec3(1.5f, 1.5f, 1.5f);
	scene.objects[i]->kappa = vec3(1.0f, 1.0f, 1.0f);
	scene.objects[i]->isReflective = false;
	scene.objects[i++]->isRefractive = true;*/

	scene.render(image);
	fullScreenTexturedQuad.Create(image);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	glUseProgram(shaderProgram); 	// make this program run
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(screenWidth, screenHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

