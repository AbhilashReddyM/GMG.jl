# Example driver for the geometric multigrid program, GMG.jl, to solve a Poisson equation on a 2D grid using 
#  second order finite difference method.
#  Dirichlet Boundary conditions


include("GMG.jl")
using .GMG: V_cycle!
using Formatting: printfmt

# Define the problem being solved. It is on a 1x1 square domain with 
# homogeneous dirichlet boundary conditions
function U_exact(x,y)
  return (x^3-x)*(y^3-y)
end

function source(x,y)
  return 6*x*y*(x^2 + y^2 -2)
end

max_cycles = 10              # max number of V cycles to execute
nlevels    = 10              # number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 
nx         = 1*2^(nlevels-1) # Nx and Ny are given as function of grid levels
ny         = 1*2^(nlevels-1) #

const u_ex = zeros(Float64,(nx+2,ny+2))
const u = zeros(Float64,(nx+2,ny+2))
const f = zeros(Float64,(nx+2,ny+2))
const res = zeros(Float64,(nx+2,ny+2))
const error = zeros(Float64,(nx,ny))

#calcualte the RHS and exact solution
DX=1.0/nx
DY=1.0/ny
xc=range(0.5*DX,1-0.5*DX,length=nx)
yc=range(0.5*DY,1-0.5*DY,length=ny)

#set the exact solution and the source term corresponding to it
for i= 1:nx, j=1:ny;
  u_ex[i+1,j+1] = U_exact(xc[i],yc[j])
  f[i+1,j+1]    = source(xc[i],yc[j])
end

println("-------------------------New Run------------------------------")
println("           Poisson equation on $(nx)X$(ny) grid")

for it=1:max_cycles

  u[:,:],res[:,:]=V_cycle!(nx,ny,nlevels,u,f,1)

  error[:,:]=u_ex[2:nx+1,2:ny+1]-u[2:nx+1,2:ny+1]

  printfmt( "|V cycle: {:2d}| res. norm: {:16.9e}| error norm: {:16.9e}|\n", it,findmax(abs.(res))[1],findmax(abs.(error))[1] )
end
