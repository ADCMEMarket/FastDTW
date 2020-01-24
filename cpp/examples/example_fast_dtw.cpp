#include "DTW.h"
#include "FastDTW.h"
#include "EuclideanDistance.h"
#include <iostream>

using namespace fastdtw;


double *sample1;
double *sample2;

std::vector<double> time_Series_1, time_Series_2;

void testDTW()
{
    TimeSeries<double,1> tsI;
    for (int i = 0; i<time_Series_1.size(); ++i) {
        tsI.addLast(i, TimeSeriesPoint<double,1>(sample1+i));
    }

    TimeSeries<double,1> tsJ;
    for (int i = 0;i<time_Series_2.size(); ++i)
    {
        tsJ.addLast(i, TimeSeriesPoint<double,1>(sample2+i));
    }

    TimeWarpInfo<double> info =  STRI::getWarpInfoBetween(tsI,tsJ,EuclideanDistance());
    printf("Warp Distance by DTW:%lf\n",info.getDistance());
    info.getPath()->print(std::cout);
}

void testFastDTW()
{
    TimeSeries<double,1> tsI;
    for (int i = 0; i<time_Series_1.size(); ++i) {
        tsI.addLast(i, TimeSeriesPoint<double,1>(sample1+i));
    }

    TimeSeries<double,1> tsJ;
    for (int i = 0;i<time_Series_2.size(); ++i)
    {
        tsJ.addLast(i, TimeSeriesPoint<double,1>(sample2+i));
    }

    TimeWarpInfo<double> info =  FAST::getWarpInfoBetween(tsI,tsJ,EuclideanDistance());
    printf("Warp Distance by DTW:%lf\n",info.getDistance());
    info.getPath()->print(std::cout);
    auto path = info.getPath();
    double cost = 0.0;
    for(int i=0;i<path->_tsIindexes.size();i++){
        long long p = path->_tsIindexes[i], q = path->_tsJindexes[i];
        printf("(%lld,%lld) --> ", p, q);
        cost += abs(sample1[p]-sample2[q]);
    }
    printf("\n cost = %f\n", cost);
}


int main(int argc , char ** argv)
{
    time_Series_1.push_back(0);
    time_Series_1.push_back(0);
    time_Series_1.push_back(0);
    time_Series_1.push_back(0);
    time_Series_1.push_back(1);
    time_Series_1.push_back(1);
    time_Series_1.push_back(2);
    time_Series_1.push_back(2);
    time_Series_1.push_back(3);
    time_Series_1.push_back(2);
    time_Series_1.push_back(1);
    time_Series_1.push_back(1);
    time_Series_1.push_back(10);
    time_Series_1.push_back(0);
    time_Series_1.push_back(0);
    time_Series_1.push_back(0);
    time_Series_1.push_back(0);

    sample1 = &time_Series_1[0];


    time_Series_2.push_back(0);
    time_Series_2.push_back(0);
    time_Series_2.push_back(1);
    time_Series_2.push_back(1);
    time_Series_2.push_back(2);
    time_Series_2.push_back(2);
    time_Series_2.push_back(3);
    time_Series_2.push_back(3);
    time_Series_2.push_back(3);
    time_Series_2.push_back(3);
    time_Series_2.push_back(2);

    sample2 = &time_Series_2[0];

    testDTW();
    testFastDTW();


    return 0;
}
