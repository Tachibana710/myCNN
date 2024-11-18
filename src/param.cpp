#include "param.hpp"

#include <string> 
#include <fstream>

void load_param(){
    std::string yaml_path = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\"));
    yaml_path += "/../param.yaml";


}