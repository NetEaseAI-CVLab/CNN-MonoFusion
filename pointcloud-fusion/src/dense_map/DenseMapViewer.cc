#include "DenseMapViewer.h"
#include<pangolin/pangolin.h>
#include<opencv2/highgui.hpp>

using namespace std;

DenseMapViewer::DenseMapViewer(DenseMapDrawer* drawer, std::string setting_file):
	m_map_drawer(drawer), m_should_finish(false), m_is_finished(false)
{
	cv::FileStorage settings(setting_file, cv::FileStorage::READ);
	m_viewpoint_x = settings["Viewer.ViewpointX_DenseMap"];
	m_viewpoint_y = settings["Viewer.ViewpointY_DenseMap"];
	m_viewpoint_z = settings["Viewer.ViewpointZ_DenseMap"];
	m_viewpoint_f = settings["Viewer.ViewpointF_DenseMap"];
	settings.release();
}



void DenseMapViewer::run()
{
	set_finished(false);
	pangolin::CreateWindowAndBind("Dense Map Viewer", 1024, 768);

	// 3D Mouse handler requires depth testing to be enabled
    //GLfloat light_position[] = { 1.0,1.0,1.0,0.0 };
    //GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
    //GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
    //GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };

    //glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    //glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    //glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);


    //glEnable(GL_LIGHTING);
    //glEnable(GL_LIGHT0);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    //glClear(GL_COLOR_BUFFER_BIT);

	// Issue specific OpenGl we might need
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Define Camera Render Object (for view / scene browsing)
	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(1024, 768, m_viewpoint_f, m_viewpoint_f, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(m_viewpoint_x, m_viewpoint_y, m_viewpoint_z,0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
	);

	// Add named OpenGL viewport to window and provide 3D Handler
	pangolin::View& d_cam = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
		.SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("dense_map_menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menu_follow_cam("dense_map_menu.FollowCamera", true, true);
    pangolin::Var<bool> menu_label_draw("dense_map_menu.Label Draw", false, true);
    pangolin::Var<bool> menu_singleframe("dense_map_menu.Single Frame", false, true);
    pangolin::Var<bool> menu_running("dense_map_menu.System Run", true, true);
    pangolin::Var<bool> menu_draw_color("dense_map_menu.Draw Color", true, true);
    pangolin::Var<bool> menu_draw_plane("dense_map_menu.Draw Plane", false, true);

    pangolin::Var<float> menu_cubesize("menu. Cube Size", 0.1, 0.01, 0.3);
    pangolin::Var<int> menu_ngrid("menu. Grid Elements", 4, 1, 10);
    pangolin::Var<float> menu_sizegrid("menu. Element Size", 0.1, 0.1, 0.3);


	pangolin::OpenGlMatrix Twc;
	Twc.SetIdentity();
    vector<Plane*> vpPlane;
	bool follow_cam = true;
    bool draw_label = true;
    bool draw_single_frame = true;
    bool draw_color = true;
    bool stop = true;
    bool draw_plane = true;
	while (true)
	{
        
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        cv::Mat cv_Tcw;
		m_map_drawer->get_current_opengl_camera_matrix(Twc, cv_Tcw);

        if(menu_label_draw)
        {
            draw_label = true;
        }
        else
        {
            draw_label = false;
        }

        if(menu_singleframe)
        {
            draw_single_frame = true;
        }
        else
        {
            draw_single_frame = false;
        }

        if (menu_draw_color)
        {
            draw_color = true;
        }
        else
        {
            draw_color = false;
        }

        // control system running
        if(menu_running)
        {
            if(!system_running)
            {
                system_running = true;
                m_map_drawer->set_system_status(true);
            }
        }
        else
        {
            if(system_running)
            {
                system_running = false; 
                m_map_drawer->set_system_status(false);
            }
        }

        
        if (menu_follow_cam && follow_cam)
		{
			s_cam.Follow(Twc);
		}
		else if (menu_follow_cam && !follow_cam)
		{
			s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(m_viewpoint_x, m_viewpoint_y, m_viewpoint_z, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0));
			s_cam.Follow(Twc);
			follow_cam = true;
		}
		else if (!menu_follow_cam && follow_cam)
		{
			follow_cam = false;
		}

		d_cam.Activate(s_cam);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vector<cv::Vec3f> points;
		m_map_drawer->draw_dense_points(points, draw_label, draw_single_frame, draw_color);

        if(menu_draw_plane)
        {
            draw_plane = true;
        }
        else
        {
            draw_plane = false;
        }

        if(draw_plane)
        {
            //Plane* pPlane = DetectPlane(cv_Tcw, points, 50);
            Plane* pPlane = DetectPlane(cv_Tcw, points, 100);
            draw_plane = false;
            menu_draw_plane = false;
            bool pushin = true;

            vector<Plane*> vpPlane2(vpPlane);
            for(int i = 0; i < vpPlane2.size(); i++)
            {
                Plane* pPlane2 = vpPlane2[i];
                //cout << pPlane2->XC << endl;
                if(norm(pPlane2->XC, pPlane->XC, CV_L2) < 0.4)
                {
                    pushin = false;
                    break;
                }
                pushin = true;
                cout << norm(pPlane2->XC, pPlane->XC, CV_L2);
            }

            if(pPlane && pushin)
            {
                cout << "New virtual cube inserted!" << endl;
                vpPlane.push_back(pPlane);
            }
            else
            {
                cout << "No plane detected. Point the camera to a planar region." << endl;
            }

        }
        if(!vpPlane.empty())
        {
            bool bRecompute = false;
            bool bLocalizationMode = true;
            if(!bLocalizationMode)
            {
                if(1)
                {
                    cout << "Map changed. All virtual elements are recomputed!" << endl;
                    bRecompute = false;
                }
            }
            for(size_t i = 0; i<vpPlane.size(); i++)
            {
                Plane* pPlane = vpPlane[i];

                if(pPlane)
                {
                    if(bRecompute)
                    {
                        pPlane->Recompute();
                    }
                    glPushMatrix();
                    pPlane->glTpw.Multiply();

                    // Draw cube
                    if(1)
                    {
                        DrawCube(menu_cubesize);
                    }

                    // Draw grid plane
                    if(1)
                    {
                        //DrawPlane(menu_ngrid, menu_sizegrid);
                    }

                    glPopMatrix();
                }
            }

        }


		pangolin::FinishFrame();

        cv::waitKey(10);

		if (should_finish())
			break;
	}

	set_finished(true);
}

void DenseMapViewer::request_finish()
{
	unique_lock<mutex> lock(m_mutex);
	m_should_finish = true;
}

bool DenseMapViewer::should_finish()
{
	unique_lock<mutex> lock(m_mutex);
	return m_should_finish;
}

bool DenseMapViewer::is_finished()
{
	unique_lock<mutex> lock(m_mutex);
	return m_is_finished;
}

void DenseMapViewer::set_finished(bool status)
{
	unique_lock<mutex> lock(m_mutex);
	m_is_finished = status;
	m_should_finish = status;
}

Plane* DenseMapViewer::DetectPlane(const cv::Mat Tcw, std::vector<cv::Vec3f> &vMPs, const int iterations)
{
    vector<cv::Mat> vPoints;
    vPoints.reserve(vMPs.size());
    vector<cv::Vec3f> vPointMP;
    vPointMP.reserve(vMPs.size());
    
    vector<cv::Vec3f> colors;


    const int M = vMPs.size();
    printf("vPoints size:%d\n", M);
    if((vMPs.empty()))
    {
        printf("error");
        return NULL;
        //return;
    }
    //for(size_t i = 0; i<vMPs.size(); i = i+ 30)
    int end = 0;
    if(vMPs.size() >= 100000)
    {
        cout << "vmp size" << vMPs.size() << endl;
        end = vMPs.size() - 100000;
    }
    
    cout << "end" << end << endl;
    for(size_t i = vMPs.size(); i > end; i = i - 60)
    {
        cv::Vec3f point = vMPs[i];

        cv::Mat img = cv::Mat(3, 1, CV_32F);
        img.at<float>(0) = point[0];
        img.at<float>(1) = point[1];
        img.at<float>(2) = point[2];
        //cout << endl << point << endl;
        //cout << endl << img << endl;
        vPoints.push_back(img);
        vPointMP.push_back(point);
    }
    
    const int N = vPoints.size();
    
    printf("vPoints1 size:%d\n", N);
    
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i = 0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    float bestDist = 1e10;
    vector<float> bestvDist;
    
    
    for(int n = 0; n<iterations; n++)
    {
        vAvailableIndices = vAllIndices;
        
        cv::Mat A(3, 4, CV_32F);
        A.col(3) = cv::Mat::ones(3, 1, CV_32F);

        // Get min set of points
        //printf("error0");
        for(short i = 0; i < 3; ++i)
        {
            
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            
            int idx = vAvailableIndices[randi];
            //printf("error1\n");
            //printf("%d\n", idx);

            //cout << endl << vPoints[idx] << endl;
            //cout << endl << A.row(i).colRange(0, 3) << endl;
            A.row(i).colRange(0, 3) = vPoints[idx].t();
            //printf("error2\n");
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
        //printf("error");
        cv::Mat u, w, vt;
        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        const float a = vt.at<float>(3, 0);
        const float b = vt.at<float>(3, 1);
        const float c = vt.at<float>(3, 2);
        const float d = vt.at<float>(3, 3);
        
        vector<float> vDistances(N, 0);

        const float f = 1.0f / sqrt(a*a + b*b + c*c + d*d);

        for(int i = 0; i<N; i++)
        {
            vDistances[i] = fabs(vPoints[i].at<float>(0)*a + vPoints[i].at<float>(1)*b + vPoints[i].at<float>(2)*c + d)*f;
        }

        vector<float> vSorted = vDistances;
        std::sort(vSorted.begin(), vSorted.end());
        //cout << N << endl;
        int nth = max((int)(0.2*N), 20);
        const float medianDist = vSorted[nth];

        if(medianDist<bestDist)
        {
            bestDist = medianDist;
            bestvDist = vDistances;
        }
    }
    
    const float th = 1.4 *bestDist;
    vector<bool> vbInliers(N, false);
    int nInliers = 0;
    for(int i = 0; i<N; i++)
    {
        if(bestvDist[i]<th)
        {
            nInliers++;
            vbInliers[i] = true;
        }
    }

    vector<cv::Vec3f> vInlierMPs(nInliers, NULL);
    int nin = 0;
    for(int i = 0; i<N; i++)
    {
        if(vbInliers[i])
        {
            vInlierMPs[nin] = vPointMP[i];
            nin++;
        }
    }
    //return;
    return new Plane(vInlierMPs, Tcw);
}

Plane::Plane(const std::vector<cv::Vec3f> &vMPs, const cv::Mat &Tcw) :mvMPs(vMPs), mTcw(Tcw.clone())
{
    rang = -3.14f / 2 + ((float)rand() / RAND_MAX)*3.14f;
    Recompute();
}

const float eps = 1e-4;
cv::Mat ExpSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
    const float d2 = x*x + y*y + z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3, 3) << 0, -z, y,
        z, 0, -x,
        -y, x, 0);
    if(d<eps)
        return (I + W + 0.5f*W*W);
    else
        return (I + W*sin(d) / d + W*W*(1.0f - cos(d)) / d2);
}

cv::Mat ExpSO3(const cv::Mat &v)
{
    return ExpSO3(v.at<float>(0), v.at<float>(1), v.at<float>(2));
}

void Plane::Recompute()
{
    const int N = mvMPs.size();

    if(mvMPs.empty())
    {
        return;
    }

    // Recompute plane with all points
    cv::Mat A = cv::Mat(N, 4, CV_32F);
    A.col(3) = cv::Mat::ones(N, 1, CV_32F);

    o = cv::Mat::zeros(3, 1, CV_32F);

    /*
    for(size_t i = 0; i<vMPs.size(); i++)
    {
        cv::Vec3f point = vMPs[i];

        cv::Mat img = cv::Mat(3, 1, CV_32F);
        img.at<float>(0) = point[0];
        img.at<float>(1) = point[1];
        img.at<float>(2) = point[2];
        //cout << endl << point << endl;
        //cout << endl << img << endl;
        vPoints.push_back(img);
        vPointMP.push_back(point);
    }
    */
    int nPoints = 0;
    for(int i = 0; i<N; i++)
    {
        cv::Vec3f point = mvMPs[i];
        cv::Mat img = cv::Mat(3, 1, CV_32F);
        img.at<float>(0) = point[0];
        img.at<float>(1) = point[1];
        img.at<float>(2) = point[2];
        cv::Mat Xw = img;
        o += Xw;
        A.row(nPoints).colRange(0, 3) = Xw.t();
        nPoints++;
    }
    A.resize(nPoints);

    cv::Mat u, w, vt;
    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    float a = vt.at<float>(3, 0);
    float b = vt.at<float>(3, 1);
    float c = vt.at<float>(3, 2);

    o = o*(1.0f / nPoints);
    const float f = 1.0f / sqrt(a*a + b*b + c*c);

    // Compute XC just the first time
    if(XC.empty())
    {
        cv::Mat Oc = -mTcw.colRange(0, 3).rowRange(0, 3).t()*mTcw.rowRange(0, 3).col(3);
        XC = Oc - o;
    }
    //cout << XC << endl;
    //cout << o << endl;
    if((XC.at<float>(0)*a + XC.at<float>(1)*b + XC.at<float>(2)*c)>0)
    {
        a = -a;
        b = -b;
        c = -c;
    }

    const float nx = a*f;
    const float ny = b*f;
    const float nz = c*f;

    n = (cv::Mat_<float>(3, 1) << nx, ny, nz);

    cv::Mat up = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 0.0f);

    cv::Mat v = up.cross(n);
    const float sa = cv::norm(v);
    const float ca = up.dot(n);
    const float ang = atan2(sa, ca);
    Tpw = cv::Mat::eye(4, 4, CV_32F);


    Tpw.rowRange(0, 3).colRange(0, 3) = ExpSO3(v*ang / sa)*ExpSO3(up*rang);
    o.copyTo(Tpw.col(3).rowRange(0, 3));

    glTpw.m[0] = Tpw.at<float>(0, 0);
    glTpw.m[1] = Tpw.at<float>(1, 0);
    glTpw.m[2] = Tpw.at<float>(2, 0);
    glTpw.m[3] = 0.0;

    glTpw.m[4] = Tpw.at<float>(0, 1);
    glTpw.m[5] = Tpw.at<float>(1, 1);
    glTpw.m[6] = Tpw.at<float>(2, 1);
    glTpw.m[7] = 0.0;

    glTpw.m[8] = Tpw.at<float>(0, 2);
    glTpw.m[9] = Tpw.at<float>(1, 2);
    glTpw.m[10] = Tpw.at<float>(2, 2);
    glTpw.m[11] = 0.0;

    glTpw.m[12] = Tpw.at<float>(0, 3);
    glTpw.m[13] = Tpw.at<float>(1, 3);
    glTpw.m[14] = Tpw.at<float>(2, 3);
    glTpw.m[15] = 1.0;

}




Plane::Plane(const float &nx, const float &ny, const float &nz, const float &ox, const float &oy, const float &oz)
{
    n = (cv::Mat_<float>(3, 1) << nx, ny, nz);
    o = (cv::Mat_<float>(3, 1) << ox, oy, oz);

    cv::Mat up = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 0.0f);

    cv::Mat v = up.cross(n);
    const float s = cv::norm(v);
    const float c = up.dot(n);
    const float a = atan2(s, c);
    Tpw = cv::Mat::eye(4, 4, CV_32F);
    const float rang = -3.14f / 2 + ((float)rand() / RAND_MAX)*3.14f;
    cout << rang;
    Tpw.rowRange(0, 3).colRange(0, 3) = ExpSO3(v*a / s)*ExpSO3(up*rang);
    o.copyTo(Tpw.col(3).rowRange(0, 3));

    glTpw.m[0] = Tpw.at<float>(0, 0);
    glTpw.m[1] = Tpw.at<float>(1, 0);
    glTpw.m[2] = Tpw.at<float>(2, 0);
    glTpw.m[3] = 0.0;

    glTpw.m[4] = Tpw.at<float>(0, 1);
    glTpw.m[5] = Tpw.at<float>(1, 1);
    glTpw.m[6] = Tpw.at<float>(2, 1);
    glTpw.m[7] = 0.0;

    glTpw.m[8] = Tpw.at<float>(0, 2);
    glTpw.m[9] = Tpw.at<float>(1, 2);
    glTpw.m[10] = Tpw.at<float>(2, 2);
    glTpw.m[11] = 0.0;

    glTpw.m[12] = Tpw.at<float>(0, 3);
    glTpw.m[13] = Tpw.at<float>(1, 3);
    glTpw.m[14] = Tpw.at<float>(2, 3);
    glTpw.m[15] = 1.0;
}

void DenseMapViewer::DrawCube(const float &size, const float x, const float y, const float z)
{
    //cout << x << y << z << endl;
    
    pangolin::OpenGlMatrix M = pangolin::OpenGlMatrix::Translate(-x, -size - y, -z);
    glPushMatrix();
    M.Multiply();
    pangolin::glDrawColouredCube(-size, size);
    glPopMatrix();
}

void DenseMapViewer::DrawPlane(Plane *pPlane, int ndivs, float ndivsize)
{
    glPushMatrix();
    pPlane->glTpw.Multiply();
    DrawPlane(ndivs, ndivsize);
    glPopMatrix();
}

void DenseMapViewer::DrawPlane(int ndivs, float ndivsize)
{
    // Plane parallel to x-z at origin with normal -y
    const float minx = -ndivs*ndivsize;
    const float minz = -ndivs*ndivsize;
    const float maxx = ndivs*ndivsize;
    const float maxz = ndivs*ndivsize;


    glLineWidth(2);
    glColor3f(0.7f, 0.7f, 1.0f);
    glBegin(GL_LINES);

    for(int n = 0; n <= 2 * ndivs; n++)
    {
        glVertex3f(minx + ndivsize*n, 0, minz);
        glVertex3f(minx + ndivsize*n, 0, maxz);
        glVertex3f(minx, 0, minz + ndivsize*n);
        glVertex3f(maxx, 0, minz + ndivsize*n);
    }

    glEnd();

}