/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */
#ifdef __APPLE__
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

#define WIDTH 1024
#define HEIGHT 768

template<typename T> struct gl;

template<> struct gl<unsigned char> {
	static const int format = GL_LUMINANCE;
	static const int type = GL_UNSIGNED_BYTE;
};

template<> struct gl<uchar3> {
	static const int format = GL_RGB;
	static const int type = GL_UNSIGNED_BYTE;
};

template<> struct gl<uchar4> {
	static const int format = GL_RGBA;
	static const int type = GL_UNSIGNED_BYTE;
};

template<> struct gl<float> {
	static const int format = GL_LUMINANCE;
	static const int type = GL_FLOAT;
};
template<> struct gl<uint16_t> {
	static const int format = GL_LUMINANCE;
	static const int type = GL_UNSIGNED_SHORT;
};
template<> struct gl<float3> {
	static const int format = GL_RGB;
	static const int type = GL_FLOAT;
};

template<typename T>
void drawit(T* scene, uint2 size) {
	static uint2 lastsize = { 0, 0 };
	char * t = (char*) "toto";
	int g = 1;
#ifdef SYCL
	if ((uint)lastsize.x() != (uint)size.x() ||
      (uint)lastsize.y() != (uint)size.y()) {
#else
	if (lastsize.x != size.x || lastsize.y != size.y) {
#endif
		lastsize = size;
		glutInit(&g, &t);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

		glutInitWindowPosition(100, 100);
#ifdef SYCL
		glutInitWindowSize(size.x(), size.y());
#else
		glutInitWindowSize(size.x, size.y);
#endif
		glutCreateWindow(" ");
	}

	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(-1, 1);
	glPixelZoom(1, -1);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#ifdef SYCL
	glPixelStorei(GL_UNPACK_ROW_LENGTH, size.x());
	glDrawPixels(size.x(), size.y(), gl<T>::format, gl<T>::type, scene);
#else
	glPixelStorei(GL_UNPACK_ROW_LENGTH, size.x);
	glDrawPixels(size.x, size.y, gl<T>::format, gl<T>::type, scene);
#endif
	
	glutSwapBuffers();

}
template<typename A, typename B, typename C, typename D, typename E>
void drawthem(A* scene1, B* scene2, C* scene3, D* scene4, E*, uint2 size) {
	static uint2 lastsize = { 0, 0 };
	char * t = (char*) "toto";
	int g = 1;
#ifdef SYCL
	if ((uint)lastsize.x() != (uint)size.x() ||
      (uint)lastsize.y() != (uint)size.y()) {
#else
	if (lastsize.x != size.x || lastsize.y != size.y) {
#endif
		lastsize = size;
		glutInit(&g, &t);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
		glutInitWindowSize(320 * 2, 240 * 2);
		// glutInitWindowPosition(100, 100);

		glutCreateWindow("Kfusion Display");
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#ifdef SYCL
		glPixelStorei(GL_UNPACK_ROW_LENGTH, size.x());
#else
		glPixelStorei(GL_UNPACK_ROW_LENGTH, size.x);
#endif		
		glMatrixMode(GL_PROJECTION);

		gluOrtho2D(0.0, (GLfloat) 640, 0.0, (GLfloat) 480);
		glMatrixMode(GL_MODELVIEW);

	}
	glClear(GL_COLOR_BUFFER_BIT);

	glRasterPos2i(0, 480);
#ifdef SYCL
	glPixelZoom(320.0 / size.x(), -240.0 / size.y());
	glDrawPixels(size.x(), size.y(), gl<A>::format, gl<A>::type, scene1);
	glRasterPos2i(320, 480);
	glDrawPixels(size.x(), size.y(), gl<B>::format, gl<B>::type, scene2);
	glRasterPos2i(0, 240);
	glDrawPixels(size.x(), size.y(), gl<C>::format, gl<C>::type, scene3);
	glRasterPos2i(320, 240);
	glDrawPixels(size.x(), size.y(), gl<D>::format, gl<D>::type, scene4);
#else
	glPixelZoom(320.0 / size.x, -240.0 / size.y);
	glDrawPixels(size.x, size.y, gl<A>::format, gl<A>::type, scene1);
	glRasterPos2i(320, 480);
	glDrawPixels(size.x, size.y, gl<B>::format, gl<B>::type, scene2);
	glRasterPos2i(0, 240);
	glDrawPixels(size.x, size.y, gl<C>::format, gl<C>::type, scene3);
	glRasterPos2i(320, 240);
	glDrawPixels(size.x, size.y, gl<D>::format, gl<D>::type, scene4);
#endif	
	glutSwapBuffers();

}

