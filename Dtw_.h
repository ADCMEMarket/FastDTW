// #include <vector>
// #include <algorithm>
// #include <limits>
// #include <tuple>
#include "DTW.h"
#include "FastDTW.h"
#include "EuclideanDistance.h"
#include <iostream>
using std::vector;
using std::tuple;
using namespace fastdtw;
class DTW{
  private:
    TimeSeries<double, 1> tsI, tsJ;
    vector<int> px, py;
    double cost;
  public:
    DTW(const double *s1, int n1, const double *s2, int n2, int use_fast){
        for (int i = 0; i<n1; ++i) {
            tsI.addLast(i, TimeSeriesPoint<double,1>(s1+i));
        }
        for (int i = 0; i<n2; ++i) {
            tsJ.addLast(i, TimeSeriesPoint<double,1>(s2+i));
        }
        if (use_fast){
            TimeWarpInfo<double> info =  FAST::getWarpInfoBetween(tsI,tsJ,EuclideanDistance());
            cost = info.getDistance();
            auto path = info.getPath();
            for(int i=0;i<path->_tsIindexes.size();i++){
                int p = path->_tsIindexes[i], q = path->_tsJindexes[i];
                // printf("(%lld,%lld) --> ", p, q);
                px.push_back(p); py.push_back(q);
            }
        }
        else{
            TimeWarpInfo<double> info =  STRI::getWarpInfoBetween(tsI,tsJ,EuclideanDistance());
            cost = info.getDistance();
            auto path = info.getPath();
            for(int i=0;i<path->_tsIindexes.size();i++){
                int p = path->_tsIindexes[i], q = path->_tsJindexes[i];
                // printf("(%lld,%lld) --> ", p, q);
                px.push_back(p); py.push_back(q);
            }
        }
        

        // printf("Warp Distance by DTW:%lf\n",info.getDistance());
        
        
    }

    void forward(tensorflow::OpKernelContext* context){
      tensorflow::Tensor *l, *p;
      tensorflow::TensorShape l_shape({});
      OP_REQUIRES_OK(context, context->allocate_output(0, l_shape, &l));
      auto l_tensor = l->flat<double>().data();   
      *l_tensor = cost;

      tensorflow::TensorShape p_shape({(int)(px.size()),2});
      OP_REQUIRES_OK(context, context->allocate_output(1, p_shape, &p));
      auto p_tensor = p->flat<tensorflow::int32>().data();   

      for(int i=0;i<px.size();i++){
        p_tensor[2*i] = px[i]; p_tensor[2*i+1] = py[i];
      }
    }  
};

void backward(
  double *grad_s1, double *grad_s2,
  const double *grad_l, 
  const double *s1, int n1, const double *s2, int n2, const int *p, int np){
    for(int i=0;i<n1;i++) grad_s1[i] = 0.0;
    for(int i=0;i<n2;i++) grad_s2[i] = 0.0;
    for(int k=0;k<np;k++){
      int i = p[2*k], j = p[2*k+1];
      grad_s1[i] += (s1[i]-s2[j])>0.0?grad_l[0]:-grad_l[0];
      grad_s2[j] += (s2[j]-s1[i])>0.0?grad_l[0]:-grad_l[0];
    }
}

