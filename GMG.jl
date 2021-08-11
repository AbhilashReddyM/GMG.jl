module GMG
export V_cycle!
function jacobi_relaxation!(level::Int,nx::Int,ny::Int,u::Array{Float64,2},f::Array{Float64,2},iters::Int,pre::Bool)

    dx=1.0/nx
    dy=1.0/ny
    Ax=1/dx^2;Ay=1/dy^2
    Ap=1.0/(2.0*(Ax+Ay))

    #Dirichlet BC
    u[  1,:]=-u[    2,:]
    u[end,:]=-u[end-1,:]
    u[  :,1]=-u[    :,2]
    u[:,end]=-u[:,end-1]

  #if it is a pre-sweep, u is fully zero (on the finest grid depends on BC, always true on coarse grids)
  # we can save some calculation, if doing only one iteration, which is typically the case.
    if ((pre) && (level>1))
        u[2:nx+1,2:ny+1] = -Ap*f[2:nx+1,2:ny+1]

        #Dirichlet BC
        u[  1,:]=-u[    2,:]
        u[end,:]=-u[end-1,:]
        u[  :,1]=-u[    :,2]
        u[:,end]=-u[:,end-1]
        iters=iters-1
    end

    for it = 1:iters
        u[2:nx+1,2:ny+1] = Ap*(Ax*(u[3:nx+2,2:ny+1] + u[1:nx,2:ny+1])
                             + Ay*(u[2:nx+1,3:ny+2] + u[2:nx+1,1:ny])
                             - f[2:nx+1,2:ny+1])
        #Dirichlet BC
        u[  1,:]=-u[    2,:]
        u[end,:]=-u[end-1,:]
        u[  :,1]=-u[    :,2]
        u[:,end]=-u[:,end-1]
    end
    #  if(not pre):
    #    return u,None
    
    res=zeros(Float64,nx+2,ny+2)
    res[2:nx+1,2:ny+1]=f[2:nx+1,2:ny+1]-(( Ax*(u[3:nx+2,2:ny+1]+u[1:nx,2:ny+1])
                                         + Ay*(u[2:nx+1,3:ny+2]+u[2:nx+1,1:ny])
                                         - 2.0*(Ax+Ay)*u[2:nx+1,2:ny+1]))
    return u,res
end


function restrict(nx,ny,v)
  """
  restrict 'v' to the coarser grid
  """
  v_c=zeros(Float64,nx+2,ny+2)

  v_c[2:nx+1,2:ny+1]=0.25*(v[2:2:2*nx,2:2:2*ny]+v[2:2:2*nx,3:2:2*ny+1]+v[3:2:2*nx+1,2:2:2*ny]+v[3:2:2*nx+1,3:2:2*ny+1])

  return v_c
end

function prolong(nx,ny,v)
  """
  interpolate 'v' to the fine grid
  """
  v_f=zeros(Float64,2*nx+2,2*ny+2)

  a=0.5625; b=0.1875; c= 0.0625

  v_f[2:2:2*nx  ,2:2:2*ny  ] = a*v[2:nx+1,2:ny+1]+b*(v[1:nx  ,2:ny+1]+v[2:nx+1,1:ny]  )+c*v[1:nx  ,1:ny  ]
  v_f[3:2:2*nx+1,2:2:2*ny  ] = a*v[2:nx+1,2:ny+1]+b*(v[3:nx+2,2:ny+1]+v[2:nx+1,1:ny]  )+c*v[3:nx+2,1:ny  ]
  v_f[2:2:2*nx  ,3:2:2*ny+1] = a*v[2:nx+1,2:ny+1]+b*(v[1:nx  ,2:ny+1]+v[2:nx+1,3:ny+2])+c*v[1:nx  ,3:ny+2]
  v_f[3:2:2*nx+1,3:2:2*ny+1] = a*v[2:nx+1,2:ny+1]+b*(v[3:nx+2,2:ny+1]+v[2:nx+1,3:ny+2])+c*v[3:nx+2,3:ny+2]

  return v_f
end

function V_cycle!(nx::Int,ny::Int,num_levels::Int,u::Array{Float64,2},f::Array{Float64,2},level::Int)
  """
  V cycle
  """
  if (level==num_levels)#bottom solve
    u,res=jacobi_relaxation!(level,nx,ny,u,f,50,true)
    return u,res
  end
  #Step 1: Relax Au=f on this grid
  u,res=jacobi_relaxation!(level,nx,ny,u,f,2,true)

  #Step 2: Restrict residual to coarse grid
  res_c=restrict(nx÷2,ny÷2,res)

  #Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
  e_c=zero(res_c)
  e_c,res_c=V_cycle!(nx÷2,ny÷2,num_levels,e_c,res_c,level+1)

  #Step 4: Interpolate(prolong) e_c to fine grid and add to u
  u=u+prolong(nx÷2,ny÷2,e_c)
  
  #Step 5: Relax Au=f on this grid
  u,res=jacobi_relaxation!(level,nx,ny,u,f,1,false)
  return u,res
end

end