#pragma once
#include <opencv2\core.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include <string>


class LabelClassesDict
{
public:
    LabelClassesDict()
    {
    }

    void settingClassesName(std::string classes_file)
    {
        //cv::FileStorage settings(setting_file, cv::FileStorage::READ);
        std::ifstream infile(classes_file);
        std::string line;

        std::string class_name;
        int label_value;
        while(getline(infile, line))
        {
            std::istringstream istring(line);
            istring >> class_name >> label_value;
            map_label_classes[class_name] = label_value;
        }
    }

    bool isLabelExist(std::string class_name) const
    {
        if(map_label_classes.count(class_name)) return true;
        else return false;
    }

    bool getLabelValue(std::string class_name, int &label_value) const
    {
        auto iter = map_label_classes.find(class_name);
        if(iter != map_label_classes.end())
        {
            label_value = iter->second;
            return true;
        }

        return false;
    }

private:
    std::map<std::string, int> map_label_classes;
};

namespace label_name
{
enum LabelClassesType
{
    wall = 0,
    floor = 1,
    ceiling = 2,
    bed = 3,
    window = 4,
    cabinet = 5,
    person = 6,
    door = 7,
    table = 8,
    plant = 9,
    curtain = 10,
    chair = 11,
    painting = 12,
    sofa = 13,
    shelf = 14,
    mirror = 15,
    rug = 16,
    seat = 17,
    lamp = 18,
    bathtub = 19,
    railing = 20,
    cushion = 21,
    pedestal = 22,
    box = 23,
    counter = 24,
    sink = 25,
    fireplace = 26,
    refrigerator = 27,
    stairs = 28,
    pillow = 29,
    toilet = 30,
    book = 31,
    countertop = 32,
    stove = 33,
    kitchen_island = 34,
    computer = 35,
    towel = 36,
    chandelier = 37,
    television = 38,
    clothes = 39,
    bottle = 40,
    washer = 41,
    plaything = 42,
    barrel = 43,
    basket = 44,
    bag = 45,
    cradle = 46,
    oven = 47,
    ball = 48,
    food = 49,
    pot = 50,
    animal = 51,
    bicycle = 52,
    dishwasher = 53,
    screen = 54,
    blanket = 55,
    hood = 56,
    sconce = 57,
    tray = 58,
    ashcan = 59,
    fan = 60,
    radiator = 61,
    clock = 62,
};
}
