//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
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
// Nev    : Kovacs Robert Kristof
// Neptun : R92D9T
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
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

struct FunctionValue {
public:
	float X, Y;

	FunctionValue(float x, float y){
		X = x;
		Y = y;
	}
};

class Hermit {
	FunctionValue RFunc[4] = { FunctionValue(400, 0), FunctionValue(500, -0.2), FunctionValue(600, 2.5), FunctionValue(700, 0) };
	FunctionValue GFunc[4] = { FunctionValue(400, 0), FunctionValue(450, -0.1), FunctionValue(550, 1.2), FunctionValue(700, 0) };
	FunctionValue BFunc[3] = { FunctionValue(400, 0), FunctionValue(460, 1), FunctionValue(520, 0) };

	float Calculate(float X, FunctionValue Func[], int length) {
		int i;
		for (int j = 0; j < length; j++) if (Func[j].X == X) return Func[j].Y;
		for (i = length-1; i >= 0; i--) if (Func[i].X < X) break;
		float a2 = (3 * (Func[i + 1].Y - Func[i].Y)) / powf((Func[i + 1].X - Func[i].X), 2);
		float a3 = (2 * (Func[i].Y - Func[i+1].Y)) / powf((Func[i + 1].X - Func[i].X), 3);
		return a3 * powf((X - Func[i].X), 3) + a2 * powf((X - Func[i].X), 2) + Func[i].Y;
	}

public:
	float GetRV(float X) {
		return Calculate(X, RFunc, 4);
	}

	float GetGV(float X) {
		return Calculate(X, GFunc, 4);
	}

	float GetBV(float X){
		return Calculate(X, BFunc, 3);
	}

	float GetSV(float X, float v) {
		FunctionValue SFunc[3] = { FunctionValue(150, 0), FunctionValue(450, 1000), FunctionValue(1600, 100) };
		for (int i = 0; i < 3; i++) SFunc[i].X *= (1.0f + v /18.9f);
		return Calculate(X, SFunc, 3);
	}

};

struct Hit {
	float t;
	vec3 position;
	vec3 color;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

struct Star {
	float radius;
	vec3 dir;
	float D0;
	int time;
	vec3 pt;
	float v;
	vec3 color;

	void CalculateColor() {
		float colors[3] = { 0,0,0 };
		float sVal;
		Hermit h;
		for (int i = 400; i <= 700; i++)
		{
			sVal = h.GetSV(i, v);
			colors[0] += sVal * h.GetRV(i);
			colors[1] += sVal * h.GetGV(i);
			colors[2] += sVal * h.GetBV(i);
		}
		color = vec3(colors[0], colors[1], colors[2]);
	}

public:
	Star(const vec3& p0, float _radius) {
		radius = _radius;
		time = 0;
		D0 = sqrtf((p0.x) * (p0.x) + (p0.y) * (p0.y) + (p0.z) * (p0.z));
		dir = p0 /D0;
		t(0);
	}
	
	vec3 getColor() {
		return color;
	}

	void normalizeColor(float v) {
		color.x /= v;
		color.y /= v;
		color.z /= v;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - pt;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.color = color;
		return hit;
	}

	void t(int T) {
		time = T;
		float Dt = D0 * powf(M_E, 0.1f * time);
		v = 0.1 * Dt;
		pt = Dt * dir;
		CalculateColor();
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

class Scene {
	std::vector<Star*> stars;
	Camera camera;
	
	float rndXY(float z) {
		int rndE = (int)rand() % 2;
		return (rndE ? (float)(rand() % ((int)z+4) * tanf(2 * M_PI / 180)) : -(float)(rand() % ((int)z+4) * tanf(2 * M_PI / 180)));
	}
	
	vec3 firstIntersect(Ray ray) {
		Hit bestHit;
		for (Star* star : stars) {
			Hit hit = star->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		return bestHit.color;
	}
	
	void detectorSetup() {
		Star* s = stars[0];
		for (int i = 0; i < 100; i++) {
			float l1 = stars[i]->getColor().x + stars[i]->getColor().y + stars[i]->getColor().z;
			float l2 = s->getColor().x + s->getColor().y + s->getColor().z;
			if (l1 > l2) s = stars[i];
		}
		float v = s->getColor().x;
		for (int i = 0; i < 100; i++) {
			stars[i]->normalizeColor(v / 5);
		}
	}

public:
	void build() {
		vec3 eye = vec3(0, 0, -28.63625), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 4 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		float dc = 3.0f;
		float dc2 = 44.5f;
		for (int i = 0; i < 100; i++) {
			float x = rndXY(i*dc+ dc2);
			float y = rndXY(i*dc+ dc2);
			stars.push_back(new Star(vec3(x, y, i*dc+ dc2), 1));
		}
	}

	void render(std::vector<vec4>& image) {
		detectorSetup();
		for (int Y = 0; Y < windowHeight; Y++) {
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = firstIntersect(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
	
	void changeTime(int T) {
		for ( int i = 0; i < 100; i++) stars[i]->t(T*2);
	}

};

GPUProgram gpuProgram;
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image): texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);	
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
bool painted = false;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	painted = true;
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

int toInt(char c) {
	switch (c)
	{
	case '0':
		return 0;
	case '1':
		return 1;
	case '2':
		return 2;
	case '3':
		return 3;
	case '4':
		return 4;
	case '5':
		return 5;
	case '6':
		return 6;
	case '7':
		return 7;
	case '8':
		return 8;
	case '9':
		return 9;
	default:
		return -1;
	}
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	int k = toInt(key);
	if (k != -1) {
		printf("%d\n", k);
		scene.changeTime(k);
		glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}