"""
History
    - get statistic of selected category's glb files
"""

#pip install trimesh

import os
import glob
from tqdm import tqdm
import numpy as np
import trimesh



if __name__ == "__main__":
    
    data_root_dir = '/opt/myspace/data/temp'   # path for objaverse
    
    categories = []   
    dir_ = sorted(glob.glob(data_root_dir+'/*'))

    for dir in dir_ :
        categories.append(dir.split('/')[-1])  #data_root_dir/category
    
    
    categories_info = {}
    
    for c in categories:
        categories_info[c] = {'x':[], 'y':[], 'z':[], 'size':[]}
        # 'axis' : [store info. of one mesh x N]

    
    for category in categories:
        dir_ = os.path.join(data_root_dir, category)
        
        sub_dir = sorted(glob.glob(dir_ +'/*'))
       
        for sub in tqdm(sub_dir):
            
            # inspect 3D bbox of one mesh
            
            vtx_x = []
            vtx_y = []
            vtx_z = []
            
            mesh = trimesh.load(sub+'/model.glb')

            keys = list(mesh.geometry.keys())   # has more than one object mesh

            for key in keys:   

                vertices = mesh.geometry[key].vertices

                for i in range(len(vertices)):
                    vtx_x.append(float(vertices[i][0]))
                    vtx_y.append(float(vertices[i][1]))
                    vtx_z.append(float(vertices[i][2]))

                        
            line_x = abs(min(vtx_x) - max(vtx_x))
            line_y = abs(min(vtx_y) - max(vtx_y))
            line_z = abs(min(vtx_z) - max(vtx_z))
            
            if line_x > 0 and line_y > 0 and line_z > 0:    # filter invalid
                categories_info[category]['x'].append(line_x)
                categories_info[category]['y'].append(line_y)
                categories_info[category]['z'].append(line_z)
                categories_info[category]['size'].append(line_x * line_y * line_z)
            
        categories_info[category]['x'] = [np.mean(categories_info[category]['x']), np.std(categories_info[category]['x'])]
        categories_info[category]['y'] = [np.mean(categories_info[category]['y']), np.std(categories_info[category]['y'])]
        categories_info[category]['z'] = [np.mean(categories_info[category]['z']), np.std(categories_info[category]['z'])]
        categories_info[category]['size'] = [np.mean(categories_info[category]['size']), np.std(categories_info[category]['size'])]
        
    # final result
    print(categories_info)
    
    import json

    with open('/opt/myspace/dhkim/GET3D/yaiverse/scale_info_objaverse.json', 'w') as f:
        json.dump(categories_info, f, indent=4)
