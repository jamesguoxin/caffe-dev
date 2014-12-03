#include <algorithm>
#include <vector>
#include <limits>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
    
    template <typename Dtype>
    Dtype EarlystopLayer<Dtype>::find_min(const std::vector<Dtype> loss_sequence) {
        Dtype result = 0;
        std::vector<Dtype> tmp(loss_sequence.begin(), loss_sequence.end());
        std::sort(tmp.begin(), tmp.end());
        result = tmp[0];
        return result;
    }
    
    template <typename Dtype>
    Dtype EarlystopLayer<Dtype>::find_median(const std::vector<Dtype> loss_sequence,
                                             const size_t k) {
        Dtype result = 0;
        std::vector<Dtype> tmp(loss_sequence.end() - k, loss_sequence.end());
        std::sort(tmp.begin(), tmp.end());
        result = tmp[k / 2];
        return result;
    }
    
    template <typename Dtype>
    Dtype EarlystopLayer<Dtype>::sum_lastk(const std::vector<Dtype> loss_sequence,
                                           const size_t k) {
        Dtype result = 0;
        for (size_t i = loss_sequence.size() - k; i < loss_sequence.size(); i++) {
            result = result + loss_sequence[i];
        }
        return result;
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
        EarlystopParameter early_param = this->layer_param_.earlystop_param();
        threshold_ = early_param.threshold();
        lamina_ = early_param.lamina();
        time_interval_ = early_param.time_interval();
        stop = false;
        median = 0;
        minimum = 0;
        sum_loss = 0;
        CHECK_EQ(bottom[0]->count(), 1) << "The input must be single Loss value";
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
        top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                        bottom[0]->height(), bottom[0]->width());
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        Dtype tmp = 0;
        top[0]->mutable_cpu_data()[0] = bottom_data[0];
        if (Caffe::phase() == Caffe::TRAIN) {
            //val_loss = 0;
            train_loss.push_back(bottom_data[0]);
            LOG(INFO) << "Size of train_loss is " << train_loss.size() << std::endl;
            //LOG(INFO) << "Passed Loss Value is " << bottom_data[0] << std::endl;
            //LOG(INFO) << "EARLYSTOP Training phase, latest train loss is " << train_loss.back() << std::endl;
            if (train_loss.size() > 0 && train_loss.size() % time_interval_ == 0) {
                LOG(INFO) << "Start CHECKING" << std::endl;
                std::ifstream tmp2("/Users/JamesGuo/Documents/MasterThesis/tmp.bin", std::ios::in | std::ios::binary);
                tmp2.read((char *) &val_loss, sizeof(Dtype));
                tmp2.close();
                LOG(INFO) << "Read Val_Loss " << val_loss << " from tmp.bin" << std::endl;
                
                minimum = EarlystopLayer<Dtype>::find_min(train_loss);
                median = EarlystopLayer<Dtype>::find_median(train_loss, time_interval_);
                sum_loss = EarlystopLayer<Dtype>::sum_lastk(train_loss, time_interval_);
                LOG(INFO) << "MINIMUM is " << minimum << std::endl;
                LOG(INFO) << "MEDIAN is " << median << std::endl;
                LOG(INFO) << "SUM is " << sum_loss << std::endl;
                tmp = time_interval_ * median / (sum_loss - time_interval_ * median);
                tmp = tmp * (val_loss - minimum) / (lamina_ * minimum);
                LOG(INFO) << "The value for comparison is " << tmp << std::endl;
                if (tmp > threshold_) {
                    stop = true;
                }
                
                if (stop == true) {
                    bottom[0]->mutable_cpu_diff()[0] = 0;
                    LOG(INFO) << "Sub-task should be terminated" << std::endl;
                } else {
                    LOG(INFO) << "Sub-task should continue" << std::endl;
                }
            }
        } else if (Caffe::phase() == Caffe::TEST) {
            //LOG(INFO) << "EARLYSTOP Testing phase" << std::endl;
            val_loss = bottom_data[0];
            std::ofstream tmp1("/Users/JamesGuo/Documents/MasterThesis/tmp.bin", std::ios::out | std::ios::binary);
            tmp1.write((char *) &val_loss, sizeof(Dtype));
            tmp1.close();
            LOG(INFO) << "Wrote Val_Loss " << val_loss << " to tmp.bin" << std::endl;
        }
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom) {
        LOG(INFO) << "Enter Backward propagation of Earlystop Layer" << std::endl;
        if (stop == true) {
            bottom[0]->mutable_cpu_diff()[0] = 0;
            LOG(INFO) << "EarlyStop Gradient is 0, update will be terminated" << std::endl;
        } else {
            bottom[0]->mutable_cpu_diff()[0] = 1;
            LOG(INFO) << "EarlyStop Gradient is 1, update will continue" << std::endl;
        }
    }
    
    INSTANTIATE_CLASS(EarlystopLayer);
    REGISTER_LAYER_CLASS(EARLYSTOP, EarlystopLayer);
} // namespace caffe
