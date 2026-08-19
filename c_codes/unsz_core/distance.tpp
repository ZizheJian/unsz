#pragma once

namespace unsz
{
    template<typename T_coord>
    T_coord get_dist_to_coord(T_coord *coord1,T_coord *coord2,uint32_t dim)
    {
        T_coord dist=0;
        for (uint32_t d=0;d<dim;d++)
            dist+=(coord1[d]-coord2[d])*(coord1[d]-coord2[d]);
        return dist;
    }

    template<typename T_coord>
    T_coord get_dist_to_block(T_coord *coord,T_coord *min_coord,T_coord *max_coord,uint32_t dim)
    {
        T_coord dist=0;
        for (uint32_t d=0;d<dim;d++)
            if (coord[d]<min_coord[d])
                dist+=(coord[d]-min_coord[d])*(coord[d]-min_coord[d]);
            else if (coord[d]>max_coord[d])
                dist+=(coord[d]-max_coord[d])*(coord[d]-max_coord[d]);
        return dist;
    }
}
