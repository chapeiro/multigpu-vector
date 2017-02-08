#include "argument_parsing.cuh"

#include <getopt.h>
#include <iostream>

using namespace std;


std::ostream& operator<<(std::ostream& out, const params_t& p){
    out << "N: " << p.N << " M: " << p.M << " Threshold: " << p.thres;
    return out;
}


int parse_args(int argc, char *argv[], params_t &params){
    int c;
    while ((c = getopt (argc, argv, "N:M:t:")) != -1){
        switch (c){
            case 'N':
                {
                    char * tmp   = optarg;
                    char * start = optarg;
                    size_t val = 1;
                    while (*tmp){
                        char t = *tmp;
                        if (t >= '0' && t <= '9'){
                            ++tmp;
                            continue;
                        } else {
                            *tmp = 0;
                            val *= atoll(start);
                            *tmp = t;
                            start = tmp + 1;

                            ++tmp;
                            if (t == '*') continue;

                            if      (t == 'K') val *= 1024;
                            else if (t == 'M') val *= 1024*1024;
                            else if (t == 'G') val *= 1024*1024*1024;
                            else {
                                cout << "Invalid entry for option -N: " << optarg << endl;
                                return -2;
                            }
                            if (*tmp) {
                                cout << "Invalid entry for option -N: " << optarg << endl;
                                return -3;
                            }
                            break;
                        }
                    }
                    if (start != tmp){
                        val *= atoll(start);
                    }
                    params.N = val;
                    break;
                }
            case 'M':
                {
                    char * tmp   = optarg;
                    char * start = optarg;
                    size_t val = 1;
                    while (*tmp){
                        char t = *tmp;
                        if (t >= '0' && t <= '9'){
                            ++tmp;
                            continue;
                        } else {
                            *tmp = 0;
                            val *= atoll(start);
                            *tmp = t;
                            start = tmp + 1;

                            ++tmp;
                            if (t == '*') continue;

                            if      (t == 'K') val *= 1024;
                            else if (t == 'M') val *= 1024*1024;
                            else if (t == 'G') val *= 1024*1024*1024;
                            else {
                                cout << "Invalid entry for option -M: " << optarg << endl;
                                return -2;
                            }
                            if (*tmp) {
                                cout << "Invalid entry for option -M: " << optarg << endl;
                                return -3;
                            }
                            break;
                        }
                    }
                    if (start != tmp){
                        val *= atoll(start);
                    }
                    params.M = val;
                    break;
                }
            case 't':
                {
                    char *end;
                    int32_t value = strtol(optarg, &end, 10); 
                    if (end == optarg || *end != '\0' || errno == ERANGE){
                        cout << "Invalid entry for option -t: " << optarg << endl;
                        return -2;
                    }
                    params.thres = value;
                    break;
                }
            case '?':
                if (optopt == 'N')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                return -4;
            default:
                assert(false);
                break;
        }
    }
    return 0;
}
