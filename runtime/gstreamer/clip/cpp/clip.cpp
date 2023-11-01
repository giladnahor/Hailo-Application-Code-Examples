/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <vector>
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "clip.hpp"
#include "hailo_tracker.hpp"
#include "hailo_xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

#define OUTPUT_LAYER_NAME "clip_rn50x4/conv89"

std::string tracker_name = "hailo_face_tracker";

void clip(HailoROIPtr roi, std::string layer_name)
{
    if (!roi->has_tensors())
    {
        return;
    }

    std::string jde_tracker_name = tracker_name + "_" + roi->get_stream_id();
    auto unique_ids = hailo_common::get_hailo_track_id(roi);
    // Remove previous matrices
    if(unique_ids.empty())
        roi->remove_objects_typed(HAILO_MATRIX);
    else
        HailoTracker::GetInstance().remove_matrices_from_track(jde_tracker_name, unique_ids[0]->get_id());
    // Convert the tensor to xarray.
    auto tensor = roi->get_tensor(layer_name);
    xt::xarray<float> embeddings = common::get_xtensor_float(tensor);

    // vector normalization
    auto normalized_embedding = common::vector_normalization(embeddings);

    HailoMatrixPtr hailo_matrix = hailo_common::create_matrix_ptr(normalized_embedding);
    if(unique_ids.empty())
    {
        roi->add_object(hailo_matrix);
    }
    else
    {
        // Update the tracker with the results
        HailoTracker::GetInstance().add_object_to_track(jde_tracker_name,
                                                        unique_ids[0]->get_id(),
                                                        hailo_matrix);
    }
}

void filter(HailoROIPtr roi)
{
    clip(roi, OUTPUT_LAYER_NAME);
}
