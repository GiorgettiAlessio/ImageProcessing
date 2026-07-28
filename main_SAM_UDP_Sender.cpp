//-------------------------- RINOMINARE IN main.cpp E RICOMPILARE----------------------
//cd build
//cmake ..
//make -j$(nproc)
//./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from 1 --udp-ip 127.0.0.1 --udp-port 5065
#include "fast_sam_3dbody.h"
#include "bvh_writer.h"
#include "cli_common.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Config : public CommonConfig
{
    Config() {
        from = "0";  
    }

    bool        skip_body      = false;
    bool        zero_face      = true;
    float       focal_x        = 0.f;
    float       focal_y        = 0.f;
    float       cx             = 0.f;
    float       cy             = 0.f;
    bool        info_only      = false;

    int         cap_w          = 0;     
    int         cap_h          = 0;     
    double      cap_fps        = 0.0;   

    std::string udp_ip         = "127.0.0.1";
    int         udp_port       = 5065;
};

static void print_usage(const char* prog)
{
    printf("Usage: %s [options]\n\n", prog);
    printf("  --onnx-dir PATH   Directory with ONNX files\n");
    printf("  --gguf PATH       pipeline.gguf\n");
    printf("  --yolo PATH       YOLO pose model\n");
    printf("  --from SRC        Webcam index or video path\n");
    printf("  --cuda DEVICE     CUDA device index (default 0; -1 = CPU)\n");
    printf("  --skip-body       Skip body model\n");
    printf("  --udp-ip IP       Indirizzo IP destinazione UDP (default 127.0.0.1)\n");
    printf("  --udp-port PORT   Porta UDP destinazione (default 5065)\n");
    printf("  --help / -h       This message\n");
}

static Config parse_args(int argc, char** argv)
{
    Config c;
    for (int i = 1; i < argc; ++i)
    {
        if (parse_common_arg(argc, argv, i, c)) continue;

#define ARG1(flag, field, conv) \
        if (!strcmp(argv[i], flag) && i+1 < argc) { c.field = conv(argv[++i]); continue; }
        ARG1("--udp-ip",   udp_ip,      std::string)
        ARG1("--udp-port", udp_port,    std::stoi)
#undef ARG1

        if (!strcmp(argv[i], "--size") && i+2 < argc)
        {
            c.cap_w = std::stoi(argv[++i]);
            c.cap_h = std::stoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--fps") && i+1 < argc)
        {
            c.cap_fps = std::stod(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--skip-body"))      { c.skip_body       = true;  continue; }
        if (!strcmp(argv[i], "--dev-face"))       { c.zero_face       = false; continue; }
        if (!strcmp(argv[i], "--info"))           { c.info_only       = true;  continue; }

        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(1);
    }
    return c;
}

// Strutture e bordi per il disegno dello scheletro 2D
static const int BODY_EDGES[][2] = {
    {0,1},{0,2},{1,3},{2,4},{5,6},{5,7},{7,62},{6,8},{8,41},
    {5,9},{6,10},{9,10},{9,11},{11,13},{13,15},{13,17},
    {10,12},{12,14},{14,18},{14,20},{5,69},{6,69},
};
static const int N_BODY_EDGES = (int)(sizeof(BODY_EDGES)/sizeof(BODY_EDGES[0]));

static const int RHAND_EDGES[][2] = {
    {41,24},{24,23},{23,22},{22,21},{41,28},{28,27},{27,26},{26,25},
    {41,32},{32,31},{31,30},{30,29},{41,36},{36,35},{35,34},{34,33},
    {41,40},{40,39},{39,38},{38,37},
};
static const int N_RHAND_EDGES = (int)(sizeof(RHAND_EDGES)/sizeof(RHAND_EDGES[0]));

static const int LHAND_EDGES[][2] = {
    {62,45},{45,44},{44,43},{43,42},{62,49},{49,48},{48,47},{47,46},
    {62,53},{53,52},{52,51},{51,50},{62,57},{57,56},{56,55},{55,54},
    {62,61},{61,60},{60,59},{59,58},
};
static const int N_LHAND_EDGES = (int)(sizeof(LHAND_EDGES)/sizeof(LHAND_EDGES[0]));

static void draw_skeleton_2d(cv::Mat& img, const std::vector<fsb::MHRResult>& results)
{
    for (const auto& r : results)
    {
        if (r.keypoints_2d.size() < 70*2) continue;
        const float* kp = r.keypoints_2d.data();

        auto pt = [&](int j) -> cv::Point
        {
            return { (int)kp[j*2], (int)kp[j*2+1] };
        };

        for (int e = 0; e < N_BODY_EDGES; ++e)
            cv::line(img, pt(BODY_EDGES[e][0]), pt(BODY_EDGES[e][1]), cv::Scalar(0,200,0), 2, cv::LINE_AA);

        for (int e = 0; e < N_RHAND_EDGES; ++e)
            cv::line(img, pt(RHAND_EDGES[e][0]), pt(RHAND_EDGES[e][1]), cv::Scalar(200,80,0), 1, cv::LINE_AA);

        for (int e = 0; e < N_LHAND_EDGES; ++e)
            cv::line(img, pt(LHAND_EDGES[e][0]), pt(LHAND_EDGES[e][1]), cv::Scalar(0,80,200), 1, cv::LINE_AA);

        for (int j = 0; j < 70; ++j)
        {
            cv::Scalar col;
            int r_px;
            if      (j <= 20)  { col = cv::Scalar(0,255,0);      r_px = 4; }
            else if (j <= 41)  { col = cv::Scalar(255,100,0);    r_px = 3; }
            else if (j <= 62)  { col = cv::Scalar(0,100,255);    r_px = 3; }
            else               { col = cv::Scalar(0,220,220);    r_px = 4; }
            cv::circle(img, pt(j), r_px, col, -1, cv::LINE_AA);
        }
    }
}

static int udp_open(const std::string& ip, int port, sockaddr_in& out_addr)
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        fprintf(stderr, "[udp] Impossibile creare il socket UDP.\n");
        return -1;
    }
    memset(&out_addr, 0, sizeof(out_addr));
    out_addr.sin_family = AF_INET;
    out_addr.sin_port   = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip.c_str(), &out_addr.sin_addr) <= 0)
    {
        fprintf(stderr, "[udp] Indirizzo IP non valido: %s\n", ip.c_str());
        close(sock);
        return -1;
    }
    return sock;
}

static double unix_timestamp_now()
{
    auto now = std::chrono::system_clock::now();
    auto us  = std::chrono::duration_cast<std::chrono::microseconds>(
                   now.time_since_epoch()).count();
    return (double)us / 1e6;
}

static std::vector<std::string> load_bvh_joint_names(const std::string& template_path)
{
    std::vector<std::string> names;
    std::ifstream f(template_path);
    if (!f.is_open()) f.open("./body_mhr.bvh");
    if (!f.is_open()) return names;

    std::string line;
    while (std::getline(f, line))
    {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        std::string s = line.substr(start);
        if (s.rfind("ROOT ", 0) == 0 || s.rfind("JOINT ", 0) == 0)
        {
            std::stringstream ss(s);
            std::string tag, name;
            ss >> tag >> name;
            names.push_back(name);
        }
    }
    return names;
}

static void send_pose_udp(int sock, const sockaddr_in& addr,
                          int person_id, const fsb::MHRResult& r,
                          BVHWriter& bvh_writer,
                          const std::vector<std::string>& joint_names)
{
    std::ostringstream json;
    json << "{";
    json << "\"person_id\":" << person_id << ",";
    json << "\"timestamp\":" << std::fixed << unix_timestamp_now() << ",";
 
    json << "\"unity_rotations_deg\":{";
    std::string bvh_line;
    std::vector<float> vals;
    
    if (bvh_writer.is_open() && bvh_writer.stream_frame_line(r, bvh_line))
    {
        std::stringstream ss(bvh_line);
        float v;
        while (ss >> v) vals.push_back(v);
    }

    bool first = true;
    if (!vals.empty() && !joint_names.empty())
    {
        for (size_t k = 0; k < joint_names.size(); ++k)
        {
            const auto& name = joint_names[k];
            float rx = 0.f, ry = 0.f, rz = 0.f;
            
            if (k == 0)
            {
                if (vals.size() >= 6) { rz = vals[3]; ry = vals[4]; rx = vals[5]; }
            }
            else
            {
                size_t idx = 6 + (k - 1) * 3;
                if (idx + 2 < vals.size())
                {
                    rz = vals[idx + 0];
                    ry = vals[idx + 1];
                    rx = vals[idx + 2];
                }
            }

            if (!first) json << ",";
            first = false;

            json << "\"" << name << "\":{"
                 << "\"x\":" << rx << ","
                 << "\"y\":" << ry << ","
                 << "\"z\":" << rz
                 << "}";
        }
    }
    json << "},";
 
    json << "\"joint_xyz_3d\":[";
    if (r.keypoints_3d.size() >= 70 * 3)
    {
        for (size_t j = 0; j < 70; ++j)
        {
            json << r.keypoints_3d[j*3] << "," << r.keypoints_3d[j*3+1] << "," << r.keypoints_3d[j*3+2];
            if (j + 1 < 70) json << ",";
        }
    }
    json << "],";
 
    double rx = 0.0, ry = 0.0, rz = 0.0;
    if (r.keypoints_3d.size() >= 70 * 3)
    {
        rx = (r.keypoints_3d[9*3+0] + r.keypoints_3d[10*3+0]) / 2.0;
        ry = (r.keypoints_3d[9*3+1] + r.keypoints_3d[10*3+1]) / 2.0;
        rz = (r.keypoints_3d[9*3+2] + r.keypoints_3d[10*3+2]) / 2.0;
    }
    json << "\"root_position\":{"
         << "\"x\":" << rx << ","
         << "\"y\":" << -ry << ","
         << "\"z\":" << -rz
         << "}";
 
    json << "}";
 
    std::string payload = json.str();
    sendto(sock, payload.c_str(), payload.size(), 0, (const sockaddr*)&addr, sizeof(addr));
}

int main(int argc, char** argv)
{
    Config c = parse_args(argc, argv);

    sockaddr_in udp_addr{};
    int udp_sock = udp_open(c.udp_ip, c.udp_port, udp_addr);
    if (udp_sock >= 0)
        printf("[udp] Invio pacchetti pose verso %s:%d\n", c.udp_ip.c_str(), c.udp_port);
    else {
        fprintf(stderr, "[udp] Apertura socket fallita.\n");
        return 1;
    }

    BVHWriter bvh_writer;
    std::vector<std::string> bvh_joint_names = load_bvh_joint_names(c.bvh_template);
    if (bvh_joint_names.empty())
    {
        bvh_joint_names = {"hip", "abdomen", "chest", "neck", "head", "lCollar", "lShldr", "lForeArm", "lHand", "rCollar", "rShldr", "rForeArm", "rHand", "lThigh", "lShin", "lFoot", "rThigh", "rShin", "rFoot"};
    }

    fsb::PipelineConfig pcfg;
    resolve_detector_defaults(c);
    resolve_backbone_defaults(c);
    apply_common_to_pipeline_cfg(c, pcfg);
    pcfg.skip_body_model  = c.skip_body;
    pcfg.zero_face_params = c.zero_face;

    fsb::Pipeline pipeline;
    if (!pipeline.load(pcfg))
    {
        fprintf(stderr, "[main] Pipeline load failed.\n");
        close(udp_sock);
        return 1;
    }

    if (c.info_only)
    {
        pipeline.print_info();
        pipeline.free();
        close(udp_sock);
        return 0;
    }

    cv::VideoCapture cap;
    bool is_image = false;
    bool src_is_int = !c.from.empty() && c.from.find_first_not_of("0123456789") == std::string::npos;
    bool is_webcam = src_is_int || (c.from.size() >= 10 && c.from.compare(0, 10, "/dev/video") == 0);

    if (src_is_int)
    {
        cap.open(std::stoi(c.from));
    }
    else
    {
        const char* img_exts[] = {".jpg",".jpeg",".png",".bmp",".tiff",".webp", nullptr};
        for (int k = 0; img_exts[k]; ++k)
        {
            if (c.from.size() >= strlen(img_exts[k]) &&
                c.from.compare(c.from.size() - strlen(img_exts[k]), strlen(img_exts[k]), img_exts[k]) == 0)
            {
                is_image = true;
                break;
            }
        }
        if (!is_image) cap.open(c.from);
    }

    if (!is_image && cap.isOpened())
    {
        if (c.cap_w > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH, c.cap_w);
        if (c.cap_h > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, c.cap_h);
        if (c.cap_fps > 0.0) cap.set(cv::CAP_PROP_FPS, c.cap_fps);
        if (is_webcam) cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        if (c.start_frame > 0) cap.set(cv::CAP_PROP_POS_FRAMES, (double)c.start_frame);
    }

    if (!is_image && !cap.isOpened())
    {
        fprintf(stderr, "[main] Cannot open input: %s\n", c.from.c_str());
        pipeline.free();
        close(udp_sock);
        return 1;
    }

    double source_fps = (c.cap_fps > 0.0) ? c.cap_fps : 30.0;
    std::string lbs_path = c.onnx_dir + "/body_model.lbs";
    
    bvh_writer.open(c.bvh_template, "", 1.0f / (float)source_fps, lbs_path,
                    c.bvh_body_shape_change, c.bvh_hand_shape_change,
                    c.bvh_compensate_finger_endsites, c.bvh_enforce_hand_limits,
                    c.bvh_zero_hand_pose, c.bvh_sticky_hand_pose,
                    c.bvh_rest_align, c.bvh_dump_rest_dirs);

    cv::Mat frame;
    int frame_count = 0;

    while (true)
    {
        if (is_image)
        {
            frame = cv::imread(c.from);
            if (frame.empty()) break;
        }
        else
        {
            if (!cap.read(frame) || frame.empty()) break;
        }

        auto t0 = Clock::now();
        std::vector<fsb::MHRResult> results =
            pipeline.process_bgr(frame.data, frame.cols, frame.rows);
        double inf_ms = ms_since(t0);

        // Invio UDP dei risultati
        for (int i = 0; i < (int)results.size(); ++i)
        {
            send_pose_udp(udp_sock, udp_addr, i, results[i], bvh_writer, bvh_joint_names);
        }

        frame_count++;

        // Visualizzazione OpenCV ripristinata
        cv::Mat vis = frame.clone();
        draw_skeleton_2d(vis, results);
        char hud[128];
        snprintf(hud, sizeof(hud), "frame %d | %.0f ms | %d person(s) | q=quit",
                 frame_count, inf_ms, (int)results.size());
        cv::putText(vis, hud, {10, 24},
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,255,255), 2, cv::LINE_AA);
        cv::imshow("Fast-SAM-3D-Body", vis);
        int key = cv::waitKey(is_image ? 0 : 1) & 0xFF;
        if (key == 'q' || key == 27) break;

        if (c.max_frames > 0 && frame_count >= c.max_frames) break;
        if (is_image) break;
    }

    if (bvh_writer.is_open()) bvh_writer.close();
    if (udp_sock >= 0) close(udp_sock);
    pipeline.free();
    return 0;
}