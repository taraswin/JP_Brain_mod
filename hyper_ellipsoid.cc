#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/manifold_lib.h>
 
#include <deal.II/base/quadrature_point_data.h>
 
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
 
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
 
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
 
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
 
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
 
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
 
#include <iostream>
#include <fstream>
 
#include <deal.II/base/multithread_info.h>

namespace Step44
{
  using namespace dealii;
 
  namespace Parameters
  {
 
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
 
    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");
 
        prm.declare_entry("Quadrature order",
                          "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }
 
    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order  = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }
 
 
    struct Geometry
    {
      unsigned int global_refinement;
      double       scale;
      double       outer_radius;
      double       inner_radius;
      double       cortex_thickness;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Global refinement",
                          "2",
                          Patterns::Integer(0),
                          "Global refinement level");
 
        prm.declare_entry("Grid scale",
                          "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");

        prm.declare_entry("Outer radius",
                          "2.0",
                          Patterns::Double(0.0),
                          "Outer radius");

        prm.declare_entry("Inner radius",
                          "1.0",
                          Patterns::Double(0.0),
                          "Inner radius");
                          
        prm.declare_entry("Cortex thickness",
                          "0.025",
                          Patterns::Double(0.0),
                          "Cortex thickness");
      }
      prm.leave_subsection();
    }
 
    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        global_refinement = prm.get_integer("Global refinement");
        scale             = prm.get_double("Grid scale");
        outer_radius      = prm.get_double("Outer radius");
        inner_radius      = prm.get_double("Inner radius");
        cortex_thickness  = prm.get_double("Cortex thickness");
      }
      prm.leave_subsection();
    }
 
 
    struct Materials
    {
      double nu;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    void Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Poisson's ratio",
                          "0.4999",
                          Patterns::Double(-1.0, 0.5),
                          "Poisson's ratio");
      }
      prm.leave_subsection();
    }
 
    void Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        nu = prm.get_double("Poisson's ratio");
      }
      prm.leave_subsection();
    }

    struct SimulationParameters
    {
      double gf_max;
      double gf_0;
      double exponent;
      double beta_mu;
      double mu_cortex;
      double gf_ratio;
      double gf_subcortex;
      std::string growth_type;

      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);

    };

    void SimulationParameters::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Simulation parameters");
      {
        prm.declare_entry("Growth multiplier maximum",
                          "1.6",
                          Patterns::Double(),
                          "Growth multiplier maximum");
 
        prm.declare_entry("Growth multiplier initial",
                          "1.0",
                          Patterns::Double(),
                          "Growth multiplier initial");
        
        prm.declare_entry("exponent factor",
                          "1",
                          Patterns::Double(),
                          "exponent factor");
        prm.declare_entry("stiffness ratio",
                          "3",
                          Patterns::Double(),
                          "stiffness ratio");
        prm.declare_entry("cortical shear modulus",
                          "2070",
                          Patterns::Double(),
                          "cortical shear modulus");
        prm.declare_entry("growth factor ratio",
                          "3",
                          Patterns::Double(),
                          "growth factor ratio");
        prm.declare_entry("subcortical growth factor",
                          "4.07e-4",
                          Patterns::Double(),
                          "subcortical growth factor");
        prm.declare_entry("growth type",
                          "volumetric",
                          Patterns::Selection("volumetric|differential"),
                          "type of growth");
      }
      prm.leave_subsection();
    }

    void SimulationParameters::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Simulation parameters");
      {
        gf_max = prm.get_double("Growth multiplier maximum");
        gf_0 = prm.get_double("Growth multiplier initial");
        exponent = prm.get_double("exponent factor");
        beta_mu = prm.get_double("stiffness ratio");
        mu_cortex = prm.get_double("cortical shear modulus");
        gf_ratio = prm.get_double("growth factor ratio");
        gf_subcortex = prm.get_double("subcortical growth factor");
        growth_type = prm.get("growth type");
      }
      prm.leave_subsection();
    }
 
 
    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;
      bool        use_static_condensation;
      std::string preconditioner_type;
      double      preconditioner_relaxation;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    void LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type",
                          "CG",
                          Patterns::Selection("CG|Direct"),
                          "Type of solver used to solve the linear system");
 
        prm.declare_entry("Residual",
                          "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");
 
        prm.declare_entry(
          "Max iteration multiplier",
          "1",
          Patterns::Double(0.0),
          "Linear solver iterations (multiples of the system matrix size)");
 
        prm.declare_entry("Use static condensation",
                          "true",
                          Patterns::Bool(),
                          "Solve the full block system or a reduced problem");
 
        prm.declare_entry("Preconditioner type",
                          "ssor",
                          Patterns::Selection("jacobi|ssor"),
                          "Type of preconditioner");
 
        prm.declare_entry("Preconditioner relaxation",
                          "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");
      }
      prm.leave_subsection();
    }
 
    void LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin                  = prm.get("Solver type");
        tol_lin                   = prm.get_double("Residual");
        max_iterations_lin        = prm.get_double("Max iteration multiplier");
        use_static_condensation   = prm.get_bool("Use static condensation");
        preconditioner_type       = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }
 
 
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");
 
        prm.declare_entry("Tolerance force",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");
 
        prm.declare_entry("Tolerance displacement",
                          "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }
 
    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f             = prm.get_double("Tolerance force");
        tol_u             = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }
 
 
    struct Time
    {
      double delta_t;
      double end_time;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");
 
        prm.declare_entry("Time step size",
                          "0.1",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }
 
    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t  = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }
 
    struct Multiprocessing
    {
      unsigned int threads;
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    void Multiprocessing::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Multiprocessing");
      {
        prm.declare_entry("threads", "8", Patterns::Integer(), "Thread count");
      }
      prm.leave_subsection();
    }
 
    void Multiprocessing::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Multiprocessing");
      {
        threads = prm.get_integer("threads");
      }
      prm.leave_subsection();
    }
 
    struct AllParameters : public FESystem,
                           public Geometry,
                           public Materials,
                           public SimulationParameters,
                           public LinearSolver,
                           public NonlinearSolver,
                           public Time,
                           public Multiprocessing
 
    {
      AllParameters(const std::string &input_file);
 
      static void declare_parameters(ParameterHandler &prm);
 
      void parse_parameters(ParameterHandler &prm);
    };
 
    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }
 
    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      SimulationParameters::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
      Multiprocessing::declare_parameters(prm);
    }
 
    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      SimulationParameters::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
      Multiprocessing::parse_parameters(prm);
    }
  } // namespace Parameters
 
template <int dim>
class Rotate3d
  {
     public:
       Rotate3d (const double angle,
                 const unsigned int axis)
         :
         angle(angle),
         axis(axis)
       {}
 
       Point<dim> operator() (const Point<dim> &p) const
       {
         if (axis==0)
           return Point<dim> (p(0),
                            std::cos(angle)*p(1) - std::sin(angle) * p(2),
                            std::sin(angle)*p(1) + std::cos(angle) * p(2));
         else if (axis==1)
           return Point<dim> (std::cos(angle)*p(0) + std::sin(angle) * p(2),
                            p(1),
                            -std::sin(angle)*p(0) + std::cos(angle) * p(2));
         else
           return Point<dim> (std::cos(angle)*p(0) - std::sin(angle) * p(1),
                            std::sin(angle)*p(0) + std::cos(angle) * p(1),
                            p(2));
       }
     private:
       const double angle;
       const unsigned int axis;
     };

  class Time
  {
  public:
    Time(const double time_end, const double delta_t)
      : timestep(0)
      , time_current(0.0)
      , time_end(time_end)
      , delta_t(delta_t)
    {}
 
    virtual ~Time() = default;
 
    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }
    void half_time_step()
    {
      delta_t /= 2;
    }
    void decrement()
    {
      time_current -= delta_t;
    }
 
  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    double delta_t;
  };
 
 
  template <int dim>
  class Material_Compressible_Neo_Hook_Three_Field
  {
  public:
    Material_Compressible_Neo_Hook_Three_Field(const double mu_cortex, 
                                               const double nu,
                                               const double beta_mu,
                                               const auto material_id,
                                               const double dist_from_sc,
                                               const double heaviside_function,
                                               const bool in_band)
      : det_F(1.0)
      , p_tilde(0.0)
      , J_tilde(1.0)
      , b_bar(Physics::Elasticity::StandardTensors<dim>::I)
    {
      /*mu = mu_cortex/beta_mu + (mu_cortex/beta_mu) * (beta_mu - 1) * heaviside_function;
      if(in_band && material_id==1)
        	mu=mu*1.01;*/
      {
      if (material_id == 1)
      {
        mu = mu_cortex/beta_mu;
        //if(in_band)
        //	mu=mu*1.01;
      }
      else
        mu = mu_cortex;
      }
      kappa = (2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu));
      c_1 = mu / 2.0;
      Assert(kappa > 0, ExcInternalError());
    }
 
    void update_material_data(const Tensor<2, dim> &F,
                              const double          p_tilde_in,
                              const double          J_tilde_in)
    {
      det_F                      = determinant(F);
      const Tensor<2, dim> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
      b_bar                      = Physics::Elasticity::Kinematics::b(F_bar);
      p_tilde                    = p_tilde_in;
      J_tilde                    = J_tilde_in;
 
      Assert(det_F > 0, ExcInternalError());
    }
 
    SymmetricTensor<2, dim> get_tau()
    {
      return get_tau_iso() + get_tau_vol();
    }
 
    SymmetricTensor<4, dim> get_Jc() const
    {
      return get_Jc_vol() + get_Jc_iso();
    }
 
    double get_dPsi_vol_dJ() const
    {
      return (kappa / 2.0) * (J_tilde - 1.0 / J_tilde);
    }
 
    double get_d2Psi_vol_dJ2() const
    {
      return ((kappa / 2.0) * (1.0 + 1.0 / (J_tilde * J_tilde)));
    }
 
    double get_det_F() const
    {
      return det_F;
    }
 
    double get_p_tilde() const
    {
      return p_tilde;
    }
 
    double get_J_tilde() const
    {
      return J_tilde;
    }
 
  protected:
    double kappa;
    double c_1;
    double mu;
 
    double                  det_F;
    double                  p_tilde;
    double                  J_tilde;
    SymmetricTensor<2, dim> b_bar;
 
    SymmetricTensor<2, dim> get_tau_vol() const
    {
      return p_tilde * det_F * Physics::Elasticity::StandardTensors<dim>::I;
    }
 
    SymmetricTensor<2, dim> get_tau_iso() const
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_bar();
    }
 
    SymmetricTensor<2, dim> get_tau_bar() const
    {
      return 2.0 * c_1 * b_bar;
    }
 
    SymmetricTensor<4, dim> get_Jc_vol() const
    {
      return p_tilde * det_F *
             (Physics::Elasticity::StandardTensors<dim>::IxI -
              (2.0 * Physics::Elasticity::StandardTensors<dim>::S));
    }
 
    SymmetricTensor<4, dim> get_Jc_iso() const
    {
      const SymmetricTensor<2, dim> tau_bar = get_tau_bar();
      const SymmetricTensor<2, dim> tau_iso = get_tau_iso();
      const SymmetricTensor<4, dim> tau_iso_x_I =
        outer_product(tau_iso, Physics::Elasticity::StandardTensors<dim>::I);
      const SymmetricTensor<4, dim> I_x_tau_iso =
        outer_product(Physics::Elasticity::StandardTensors<dim>::I, tau_iso);
      const SymmetricTensor<4, dim> c_bar = get_c_bar();
 
      return (2.0 / dim) * trace(tau_bar) *
               Physics::Elasticity::StandardTensors<dim>::dev_P -
             (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso) +
             Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar *
               Physics::Elasticity::StandardTensors<dim>::dev_P;
    }
 
    SymmetricTensor<4, dim> get_c_bar() const
    {
      return SymmetricTensor<4, dim>();
    }
  };
 
  template <int dim>
  class Growth
  {
    protected :
      Tensor<2,dim> F_growth;
      Tensor<2,dim> F_growth_old;
      Tensor<2,dim> radial_direction;
      Tensor<2,dim> tangential_direction;
      double gf_subcortex;
      double gf_ratio;
      double growth_multiplier;
      double v_t;
      double v_r;
      double v_g;
      double gf_0;
      double gf_max;
      double alpha;
      unsigned int exponent;
      types::material_id material_id;
      double dist_from_sc;
      double H;
      bool in_band;

    double time_factor(double time)
    {
      return (1-std::exp(-time/exponent));
    }

    public:
      Growth()
      : F_growth(Physics::Elasticity::StandardTensors<dim>::I),
        gf_max (0.0),
        exponent (0.0),
        growth_multiplier(1.0),
        v_t(1.0), v_r(1.0), v_g(1.0),
        gf_subcortex(1.0),
        dist_from_sc(0.0),
        H(0.0),
        alpha(0.5),
        in_band(false)
      {}
      virtual ~Growth() {}
      void setup (const double exponent_factor,
                  const double growth_factor_max,
                  const double subcortex_growth,
                  const double growth_ratio,
                  const Tensor<1, dim> &N,
                  const auto material,
                  const double distance,
                  const double heaviside_function,
                  const bool inside_band) 
      {
        gf_max = growth_factor_max;
        exponent = exponent_factor;
        material_id = material;
        gf_subcortex = subcortex_growth;
        gf_ratio = growth_ratio;
        radial_direction = outer_product(N,N);
        tangential_direction = Physics::Elasticity::StandardTensors<dim>::I - outer_product(N,N);
        dist_from_sc = distance;
        H = heaviside_function;
        in_band = inside_band;
      }
      /*void update_differential_growth(double time)
      {
        growth_multiplier = (gf_max-1)*0.02*time;
          v_t = 1+(std::exp(growth_multiplier)-1)*H;
          v_r = 1+(std::exp(growth_multiplier/gf_ratio)-1)*H;
          F_growth = v_t*tangential_direction + v_r*radial_direction;
      }*/
      void update_differential_growth(double time)
      {
        growth_multiplier = (gf_max-1)*time_factor(time);
        if (material_id==1)
        {
         v_g = std::sqrt(1+growth_multiplier);
         F_growth = Physics::Elasticity::StandardTensors<dim>::I;
        }
        else
        {
         v_t = std::sqrt(1+growth_multiplier);
         v_r = std::sqrt(1+growth_multiplier/gf_ratio);
         F_growth = v_t*tangential_direction + v_r*radial_direction;
        }
      }
      /*void update_differential_growth (double time)
      {
        if (dim == 2)
        {
          growth_multiplier = (gf_max-1)*time_factor(time);
          v_t = 1+ ((std::sqrt(1 + growth_multiplier) -1 )*H);
          v_r = 1+ ((std::sqrt(1 + growth_multiplier/gf_ratio) -1 )*H);
        }
        else
          growth_multiplier = std::cbrt(1+(gf_max-1)*time_factor(time));
        
        F_growth = v_t * tangential_direction + v_r * radial_direction;
      }*/
      void update_volumetric_growth (double time)
      {
        growth_multiplier = (gf_max-1)*time_factor(time);
        if(dim==2)
        v_g = 1+ ((std::sqrt(1 + growth_multiplier) -1 )*H);
        else
        v_g = 1+ ((std::cbrt(1 + growth_multiplier) -1 )*H);
        F_growth=Physics::Elasticity::StandardTensors<dim>::I*v_g;
      }      
      const Tensor<2,dim> get_growth_tensor() const
      {
        return F_growth;
      }
      double get_det_F_g() const
      {
        return determinant(F_growth);
      }
        
      double get_growth_multiplier() const
      {
        return growth_multiplier;
      }
  };

  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : F_inv(Physics::Elasticity::StandardTensors<dim>::I)
      , F_g_inv(Physics::Elasticity::StandardTensors<dim>::I)
      , tau(SymmetricTensor<2, dim>())
      , d2Psi_vol_dJ2(0.0)
      , dPsi_vol_dJ(0.0)
      , Jc(SymmetricTensor<4, dim>())
    {}
 
    virtual ~PointHistory() = default;
 
    void setup_lqp(const Parameters::AllParameters &parameters,
                   const auto material_id,
                   const Tensor<1, dim> &N,
                   const double distance,
                   const double heaviside_function)
    {
      bool in_band = false;
      Tensor<1,dim> x_axis({-1.0,0.0});
      if(std::abs(std::tan(std::acos(N*x_axis)) - 1) <= 1e-1)
      	in_band = true;
      material =
        std::make_shared<Material_Compressible_Neo_Hook_Three_Field<dim>>(
          parameters.mu_cortex, parameters.nu, parameters.beta_mu, material_id, distance, heaviside_function, in_band);
      growth = std::make_shared<Growth<dim>>();
      growth->setup(parameters.exponent, parameters.gf_max, parameters.gf_subcortex
                  , parameters.gf_ratio, N, material_id, distance, heaviside_function, in_band);
      update_values(Tensor<2, dim>(), 0.0, 1.0,
                    0.0, true, "default");
    }
 
    void update_values(const Tensor<2, dim> &Grad_u_n,
                       const double          p_tilde,
                       const double          J_tilde,
                       //const Tensor<1, dim> &N,
                       const double          time,
                       bool                  growth_update,
                       std:: string          growth_type)
    {
      if (growth_update)
      {
        if (growth_type == "volumetric")
        {
          growth->update_volumetric_growth(time);
          Tensor<2, dim> F_growth = growth->get_growth_tensor();
          F_g_inv = invert(F_growth);
        }
        else if (growth_type == "differential")
        {
          growth->update_differential_growth(time);
          Tensor<2, dim> F_growth = growth->get_growth_tensor();
          F_g_inv = invert(F_growth);
        }
        else
        {}
      }
      const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(Grad_u_n);
      Tensor<2, dim> F_elastic = F * F_g_inv;
      material->update_material_data(F_elastic, p_tilde, J_tilde);
 
      F_inv         = invert(F);
      tau           = material->get_tau();
      Jc            = material->get_Jc();
      dPsi_vol_dJ   = material->get_dPsi_vol_dJ();
      d2Psi_vol_dJ2 = material->get_d2Psi_vol_dJ2();
    }
 
    double get_J_tilde() const
    {
      return material->get_J_tilde();
    }
 
    double get_det_F() const
    {
      return material->get_det_F();
    }
 
    const Tensor<2, dim> &get_F_inv() const
    {
      return F_inv;
    }
 
    double get_p_tilde() const
    {
      return material->get_p_tilde();
    }
 
    const SymmetricTensor<2, dim> &get_tau() const
    {
      return tau;
    }
 
    double get_dPsi_vol_dJ() const
    {
      return dPsi_vol_dJ;
    }
 
    double get_d2Psi_vol_dJ2() const
    {
      return d2Psi_vol_dJ2;
    }
 
    const SymmetricTensor<4, dim> &get_Jc() const
    {
      return Jc;
    }

    double get_growth_stretch()
    {
      return growth->get_growth_multiplier();
    }
 
  private:
    std::shared_ptr<Material_Compressible_Neo_Hook_Three_Field<dim>> material;
    std::shared_ptr<Growth<dim>> growth;
 
    Tensor<2, dim> F_inv;
    Tensor<2, dim> F_g_inv;
 
    SymmetricTensor<2, dim> tau;
    double                  d2Psi_vol_dJ2;
    double                  dPsi_vol_dJ;
 
    SymmetricTensor<4, dim> Jc;
  };
 
 
 
  template <int dim>
  class Solid
  {
  public:
    Solid(const std::string &input_file);
 
    void run();
 
  private:
    struct PerTaskData_ASM;
    struct ScratchData_ASM;
 
    struct PerTaskData_SC;
    struct ScratchData_SC;
 
    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    struct PerTaskData_SQPH;
    struct ScratchData_SQPH;
 
    void make_grid();
 
    void system_setup();
 
    void determine_component_extractors();
 
    void make_constraints(const int it_nr);
 
    void assemble_system();

    void assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM                                      &scratch,
      PerTaskData_ASM                                      &data) const;
 
    void assemble_sc();
 
    void assemble_sc_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_SC                                       &scratch,
      PerTaskData_SC                                       &data);
 
    void copy_local_to_global_sc(const PerTaskData_SC &data);
 
    void setup_qph();

    void setup_qph_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_SQPH                                     &scratch,
      PerTaskData_SQPH                                     &data);

    void copy_local_to_global_SQPH(const PerTaskData_SQPH & /*data*/) {}
 
    void update_qph_incremental(const BlockVector<double> &solution_delta,
                                bool growth_update);
 
    void update_qph_incremental_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_UQPH                                     &scratch,
      PerTaskData_UQPH                                     &data);
 
    void copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}
 
    void solve_nonlinear_timestep(BlockVector<double> &solution_delta);
 
    std::pair<unsigned int, double>
    solve_linear_system(BlockVector<double> &newton_update);
 
    BlockVector<double>
    get_total_solution(const BlockVector<double> &solution_delta) const;
 
    void output_results() const;
 
    Parameters::AllParameters parameters;

    double vol_reference;
    double radius_subcortex;
 
    Triangulation<dim> triangulation;
 
    Time                time;
    mutable TimerOutput timer;
 
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim>>
      quadrature_point_history;
 
    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;
    const FEValuesExtractors::Scalar p_fe;
    const FEValuesExtractors::Scalar J_fe;
 
    static const unsigned int n_blocks          = 3;
    static const unsigned int n_components      = dim + 2;
    static const unsigned int first_u_component = 0;
    static const unsigned int p_component       = dim;
    static const unsigned int J_component       = dim + 1;
 
    enum
    {
      u_dof = 0,
      p_dof = 1,
      J_dof = 2
    };
 
    std::vector<types::global_dof_index> dofs_per_block;
    std::vector<types::global_dof_index> element_indices_u;
    std::vector<types::global_dof_index> element_indices_p;
    std::vector<types::global_dof_index> element_indices_J;
 
    const QGauss<dim>     qf_cell;
    const QGauss<dim - 1> qf_face;
    const unsigned int    n_q_points;
    const unsigned int    n_q_points_f;
 
    AffineConstraints<double> constraints;
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> tangent_matrix;
    BlockVector<double>       system_rhs;
    BlockVector<double>       solution_n;
 
    struct Errors
    {
      Errors()
        : norm(1.0)
        , u(1.0)
        , p(1.0)
        , J(1.0)
      {}
 
      void reset()
      {
        norm = 1.0;
        u    = 1.0;
        p    = 1.0;
        J    = 1.0;
      }
      void normalize(const Errors &rhs)
      {
        if (rhs.norm != 0.0)
          norm /= rhs.norm;
        if (rhs.u != 0.0)
          u /= rhs.u;
        if (rhs.p != 0.0)
          p /= rhs.p;
        if (rhs.J != 0.0)
          J /= rhs.J;
      }
 
      double norm, u, p, J;
    };
 
    Errors error_residual, error_residual_0, error_residual_norm, error_update,
      error_update_0, error_update_norm;
 
    void get_error_residual(Errors &error_residual);

    double compute_vol_current() const;
 
    void get_error_update(const BlockVector<double> &newton_update,
                          Errors                    &error_update);
 
    std::pair<double, double> get_error_dilation() const;
 
    static void print_conv_header();
 
    void print_conv_footer();

    Vector<double> new_material_ids;

    bool instability;
    unsigned int newton_iteration;
  };
 
 
 
  template <int dim>
  Solid<dim>::Solid(const std::string &input_file)
    : parameters(input_file)
    , vol_reference(0.)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , time(parameters.end_time, parameters.delta_t)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , degree(parameters.poly_degree)
    ,
    fe(FE_Q<dim>(parameters.poly_degree) ^ dim, // displacement
       FE_DGP<dim>(parameters.poly_degree - 1), // pressure
       FE_DGP<dim>(parameters.poly_degree - 1)) // dilatation
    , dof_handler(triangulation)
    , dofs_per_cell(fe.n_dofs_per_cell())
    , u_fe(first_u_component)
    , p_fe(p_component)
    , J_fe(J_component)
    , dofs_per_block(n_blocks)
    , qf_cell(parameters.quad_order)
    , qf_face(parameters.quad_order)
    , n_q_points(qf_cell.size())
    , n_q_points_f(qf_face.size())
    , instability(false)
  {
    Assert(dim == 2 || dim == 3,
           ExcMessage("This problem only works in 2 or 3 space dimensions."));
    determine_component_extractors();
  }
 
 
  template <int dim>
  void Solid<dim>::run()
  {
    MultithreadInfo::set_thread_limit(parameters.threads);
    std::cout<<parameters.growth_type<<" growth will be applied."<<std::endl;
    std::cout<<MultithreadInfo::n_cores()<<" cores are available."<<std::endl;
    std::cout<<MultithreadInfo::n_threads()<<" threads used to run this simulation."<<std::endl;
    make_grid();
    system_setup();
    {
      AffineConstraints<double> constraints;
      constraints.close();
 
      const ComponentSelectFunction<dim> J_mask(J_component, n_components);
 
      VectorTools::project(
        dof_handler, constraints, QGauss<dim>(degree + 2), J_mask, solution_n);
    }
    output_results();
    time.increment();
 
    BlockVector<double> solution_delta(dofs_per_block);
    while (time.current() < time.end())
      {
        solution_delta = 0.0;
        solve_nonlinear_timestep(solution_delta);
        solution_n += solution_delta;
        output_results();
        time.increment();
      }
  }
 
  template <int dim>
  struct Solid<dim>::PerTaskData_ASM
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
 
    PerTaskData_ASM(const unsigned int dofs_per_cell)
      : cell_matrix(dofs_per_cell, dofs_per_cell)
      , cell_rhs(dofs_per_cell)
      , local_dof_indices(dofs_per_cell)
    {}
 
    void reset()
    {
      cell_matrix = 0.0;
      cell_rhs    = 0.0;
    }
  };
 
 
  template <int dim>
  struct Solid<dim>::ScratchData_ASM
  {
    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;
 
    std::vector<std::vector<double>>                  Nx;
    std::vector<std::vector<Tensor<2, dim>>>          grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx;
 
    ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim>        &qf_cell,
                    const UpdateFlags         uf_cell,
                    const QGauss<dim - 1>    &qf_face,
                    const UpdateFlags         uf_face)
      : fe_values(fe_cell, qf_cell, uf_cell)
      , fe_face_values(fe_cell, qf_face, uf_face)
      , Nx(qf_cell.size(), std::vector<double>(fe_cell.n_dofs_per_cell()))
      , grad_Nx(qf_cell.size(),
                std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , symm_grad_Nx(qf_cell.size(),
                     std::vector<SymmetricTensor<2, dim>>(
                       fe_cell.n_dofs_per_cell()))
    {}
 
    ScratchData_ASM(const ScratchData_ASM &rhs)
      : fe_values(rhs.fe_values.get_fe(),
                  rhs.fe_values.get_quadrature(),
                  rhs.fe_values.get_update_flags())
      , fe_face_values(rhs.fe_face_values.get_fe(),
                       rhs.fe_face_values.get_quadrature(),
                       rhs.fe_face_values.get_update_flags())
      , Nx(rhs.Nx)
      , grad_Nx(rhs.grad_Nx)
      , symm_grad_Nx(rhs.symm_grad_Nx)
    {}
 
    void reset()
    {
      const unsigned int n_q_points      = Nx.size();
      const unsigned int n_dofs_per_cell = Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert(grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              Nx[q_point][k]           = 0.0;
              grad_Nx[q_point][k]      = 0.0;
              symm_grad_Nx[q_point][k] = 0.0;
            }
        }
    }
  };
 
  template <int dim>
  struct Solid<dim>::PerTaskData_SC
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;
 
    FullMatrix<double> k_orig;
    FullMatrix<double> k_pu;
    FullMatrix<double> k_pJ;
    FullMatrix<double> k_JJ;
    FullMatrix<double> k_pJ_inv;
    FullMatrix<double> k_bbar;
    FullMatrix<double> A;
    FullMatrix<double> B;
    FullMatrix<double> C;
 
    PerTaskData_SC(const unsigned int dofs_per_cell,
                   const unsigned int n_u,
                   const unsigned int n_p,
                   const unsigned int n_J)
      : cell_matrix(dofs_per_cell, dofs_per_cell)
      , local_dof_indices(dofs_per_cell)
      , k_orig(dofs_per_cell, dofs_per_cell)
      , k_pu(n_p, n_u)
      , k_pJ(n_p, n_J)
      , k_JJ(n_J, n_J)
      , k_pJ_inv(n_p, n_J)
      , k_bbar(n_u, n_u)
      , A(n_J, n_u)
      , B(n_J, n_u)
      , C(n_p, n_u)
    {}
 
    void reset()
    {}
  };
 
 
  template <int dim>
  struct Solid<dim>::ScratchData_SC
  {
    void reset()
    {}
  };
 
 
  template <int dim>
  struct Solid<dim>::PerTaskData_UQPH
  {
    void reset()
    {}
  };
 
 
  template <int dim>
  struct Solid<dim>::ScratchData_UQPH
  {
    const BlockVector<double> &solution_total;
 
    std::vector<Tensor<2, dim>> solution_grads_u_total;
    std::vector<double>         solution_values_p_total;
    std::vector<double>         solution_values_J_total;

    bool                        growth_update;
 
    FEValues<dim> fe_values;
 
    ScratchData_UQPH(const FiniteElement<dim>  &fe_cell,
                     const QGauss<dim>         &qf_cell,
                     const UpdateFlags          uf_cell,
                     const BlockVector<double> &solution_total,
                     bool growth_update)
      : solution_total(solution_total)
      , solution_grads_u_total(qf_cell.size())
      , solution_values_p_total(qf_cell.size())
      , solution_values_J_total(qf_cell.size())
      , fe_values(fe_cell, qf_cell, uf_cell)
      , growth_update(growth_update)
    {}
 
    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : solution_total(rhs.solution_total)
      , solution_grads_u_total(rhs.solution_grads_u_total)
      , solution_values_p_total(rhs.solution_values_p_total)
      , solution_values_J_total(rhs.solution_values_J_total)
      , fe_values(rhs.fe_values.get_fe(),
                  rhs.fe_values.get_quadrature(),
                  rhs.fe_values.get_update_flags())
      , growth_update(rhs.growth_update)
    {}
 
    void reset()
    {
      const unsigned int n_q_points = solution_grads_u_total.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          solution_grads_u_total[q]  = 0.0;
          solution_values_p_total[q] = 0.0;
          solution_values_J_total[q] = 0.0;
        }
    }
  };
 
  template <int dim>
  struct Solid<dim>::PerTaskData_SQPH
  {
    void reset()
    {}
  };
 
 
  template <int dim>
  struct Solid<dim>::ScratchData_SQPH
  {

    FEValues<dim>                fe_values;

    ScratchData_SQPH(const FiniteElement<dim> &fe_cell,
                     const QGauss<dim> &qf_cell,
                     const UpdateFlags uf_cell)
      :
      fe_values(fe_cell, qf_cell, uf_cell)
      {}

    ScratchData_SQPH(const ScratchData_SQPH &rhs)
      :
      fe_values(rhs.fe_values.get_fe(),
                rhs.fe_values.get_quadrature(),
                rhs.fe_values.get_update_flags())
    {}

    void reset()
    {}
  };
  
  template <int dim>
  void Solid<dim>::make_grid()
  {
  	const Point<dim> centre;
	  const Tensor<1,dim> x({1.0,0.0});
	  const double e = 0.75;
	  radius_subcortex = parameters.outer_radius - parameters.cortex_thickness;
	  // Create a hyper shell
	  GridGenerator::quarter_hyper_shell(triangulation, Point<dim>(), parameters.inner_radius, parameters.outer_radius, 0, true);
	  triangulation.refine_global(parameters.global_refinement);

	  // Define EllipticalManifold with semi-axes a, b, c
	  EllipticalManifold<dim> elliptical_manifold(centre, x, e);

	  // Set the manifold for the whole triangulation
	  triangulation.set_all_manifold_ids(0);
	  triangulation.set_manifold(0, elliptical_manifold);

	  for (const auto &cell : triangulation.active_cell_iterators())
	      {
		if (radius_subcortex - centre.distance(cell->center()) >= parameters.scale)
		{
		  cell->set_material_id(1);
		}
		else
		{
		  cell->set_material_id(2);
		}
		//material_ids(i)=static_cast<int>(cell->material_id());
		//i+=1;
	      }

	  // Apply the transformation to scale the unit ball into an ellipsoid
	  GridTools::transform(
	    [&](const Point<2> &p) -> Point<2> {
	      return Point<2>(parameters.outer_radius * p[0], parameters.outer_radius * 0.75 * p[1]);
	    },
	    triangulation);
	    
	  /*for (const auto &cell : triangulation.active_cell_iterators())
	      {
	      	for (const auto &face : cell->face_iterators())
	      	{
		    if (face->at_boundary() && (face->boundary_id() == 1))
		      {
		      	cell->set_material_id(2);
		      }
		    else
		      {
		        cell->set_material_id(1);
		      }
		}
	      }*/
	vol_reference = GridTools::volume(triangulation);
    	std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;    
  }
  
  /*template <int dim>
  void Solid<dim>::make_grid()
  {
    const Tensor<1,2> x({1.0,0.0});
    const double e = 0.75;
    const double angle = numbers::PI/2;
    radius_subcortex = parameters.outer_radius 
                      - parameters.cortex_thickness;
    GridGenerator::hyper_shell(
      triangulation, Point<dim>(),
      parameters.inner_radius, parameters.outer_radius, 0, true);
    (dim == 2) ? GridTools::rotate(angle, triangulation)
               : GridTools::transform (Rotate3d<dim>(angle, 2), triangulation);
    GridTools::scale(parameters.scale, triangulation);
    if (dim==2)
    {
      Point<dim> center_2d (0.0, 0.0);
      const SphericalManifold<dim> manifold(center_2d);
      EllipticalManifold<2> elliptical_manifold(center_2d, x, e);
      triangulation.set_all_manifold_ids_on_boundary(0);
      triangulation.set_manifold(0, elliptical_manifold);
      triangulation.refine_global(std::max (1U, parameters.global_refinement));
      //triangulation.set_manifold (0, manifold);
      Vector<double> material_ids(triangulation.n_active_cells());
      //unsigned int i = 0;
      for (const auto &cell : triangulation.active_cell_iterators())
      {
        if (radius_subcortex - center_2d.distance(cell->center())/parameters.scale >= parameters.scale)
        {
          cell->set_material_id(1);
        }
        else
        {
          cell->set_material_id(2);
        }
        //material_ids(i)=static_cast<int>(cell->material_id());
        //i+=1;
      }
      //new_material_ids = material_ids;
      GridTools::transform(
	    [&](const Point<2> &p) -> Point<2> {
	      return Point<2>(parameters.outer_radius * p[0], 0.85 * parameters.outer_radius * p[1]);
	    },
	    triangulation);
    }
    else
    {
      Point<dim> center_3d (0.0, 0.0, 0.0);
      const SphericalManifold<dim> manifold(center_3d);
      triangulation.set_all_manifold_ids_on_boundary(0);
      triangulation.refine_global(std::max (1U, parameters.global_refinement));
      triangulation.set_manifold (0, manifold);
      Vector<double> material_ids(triangulation.n_active_cells());
      unsigned int i = 0;
      for (const auto &cell : triangulation.active_cell_iterators())
      {
      if (radius_subcortex
            -center_3d.distance(cell->center())/parameters.scale >= parameters.scale)
          {
              cell->set_material_id(1);
          }
      else
          {
            cell->set_material_id(2);
          }
      //material_ids(i)=static_cast<int>(cell->material_id());
      //i+=1;
      }
      //new_material_ids = material_ids;
    }
    vol_reference = GridTools::volume(triangulation);
    std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
  }*/
 
 
 
  template <int dim>
  void Solid<dim>::system_setup()
  {
    timer.enter_subsection("Setup system");
 
    std::vector<unsigned int> block_component(n_components,
                                              u_dof); // Displacement
    block_component[p_component] = p_dof;             // Pressure
    block_component[J_component] = J_dof;             // Dilatation
 
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);
    DoFRenumbering::component_wise(dof_handler, block_component);
 
    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
 
    std::cout << "Triangulation:"
              << "\n\t Number of active cells: "
              << triangulation.n_active_cells()
              << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
 
    tangent_matrix.clear();
    {
      const types::global_dof_index n_dofs_u = dofs_per_block[u_dof];
      const types::global_dof_index n_dofs_p = dofs_per_block[p_dof];
      const types::global_dof_index n_dofs_J = dofs_per_block[J_dof];

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

      dsp.block(u_dof, u_dof).reinit(n_dofs_u, n_dofs_u);
      dsp.block(u_dof, p_dof).reinit(n_dofs_u, n_dofs_p);
      dsp.block(u_dof, J_dof).reinit(n_dofs_u, n_dofs_J);

      dsp.block(p_dof, u_dof).reinit(n_dofs_p, n_dofs_u);
      dsp.block(p_dof, p_dof).reinit(n_dofs_p, n_dofs_p);
      dsp.block(p_dof, J_dof).reinit(n_dofs_p, n_dofs_J);

      dsp.block(J_dof, u_dof).reinit(n_dofs_J, n_dofs_u);
      dsp.block(J_dof, p_dof).reinit(n_dofs_J, n_dofs_p);
      dsp.block(J_dof, J_dof).reinit(n_dofs_J, n_dofs_J);
      dsp.collect_sizes();

 
      Table<2, DoFTools::Coupling> coupling(n_components, n_components);
      for (unsigned int ii = 0; ii < n_components; ++ii)
        for (unsigned int jj = 0; jj < n_components; ++jj)
          if (((ii < p_component) && (jj == J_component)) ||
              ((ii == J_component) && (jj < p_component)) ||
              ((ii == p_component) && (jj == p_component)))
            coupling[ii][jj] = DoFTools::none;
          else
            coupling[ii][jj] = DoFTools::always;
      DoFTools::make_sparsity_pattern(
        dof_handler, coupling, dsp, constraints, false);
      sparsity_pattern.copy_from(dsp);
    }
 
    tangent_matrix.reinit(sparsity_pattern);
 
    system_rhs.reinit(dofs_per_block);
    system_rhs.collect_sizes();

    solution_n.reinit(dofs_per_block);
    solution_n.collect_sizes();
 
    setup_qph();
 
    timer.leave_subsection();
  }
 
 
  template <int dim>
  void Solid<dim>::determine_component_extractors()
  {
    element_indices_u.clear();
    element_indices_p.clear();
    element_indices_J.clear();
 
    for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k)
      {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        if (k_group == u_dof)
          element_indices_u.push_back(k);
        else if (k_group == p_dof)
          element_indices_p.push_back(k);
        else if (k_group == J_dof)
          element_indices_J.push_back(k);
        else
          {
            Assert(k_group <= J_dof, ExcInternalError());
          }
      }
  }
 
  template <int dim>
  void Solid<dim>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;
 
    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points);

    const UpdateFlags uf_SQPH(update_quadrature_points);
    PerTaskData_SQPH per_task_data_SQPH;
    ScratchData_SQPH scratch_data_SQPH(fe, qf_cell, uf_SQPH);


    WorkStream::run(dof_handler.active_cell_iterators(),
                    *this,
                    &Solid::setup_qph_one_cell,
                    &Solid::copy_local_to_global_SQPH,
                    scratch_data_SQPH,
                    per_task_data_SQPH);

  }

  template<int dim>
  void Solid<dim>::setup_qph_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_SQPH                                     &scratch,
    PerTaskData_SQPH & /*data*/)
    {
      const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

      scratch.reset();
      scratch.fe_values.reinit(cell);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        Tensor<1,dim> outward_normal = scratch.fe_values.quadrature_point(q_point);
        double dist_from_sc = outward_normal.norm()/parameters.scale - radius_subcortex;
        unsigned int material_id;
        if (outward_normal.norm()/parameters.scale >= radius_subcortex)
	  material_id = 2;
        else
	  material_id = 1;
        outward_normal /= (outward_normal.norm());
        double heaviside_function = std::exp(40*(dist_from_sc))/(1+std::exp(40*(dist_from_sc)));
        lqph[q_point]->setup_lqp(parameters, cell->material_id(), outward_normal, dist_from_sc, heaviside_function);
      }
    }
 
  template <int dim>
  void
  Solid<dim>::update_qph_incremental(const BlockVector<double> &solution_delta,
                                     bool                       growth_update)
  {
    timer.enter_subsection("Update QPH data");
    if (!growth_update)
      std::cout << " UQPH " << std::flush;
 
    const BlockVector<double> solution_total(
      get_total_solution(solution_delta));
 
    const UpdateFlags uf_UQPH(update_values | update_gradients | update_quadrature_points);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(fe, qf_cell, uf_UQPH, solution_total,
                                        growth_update);
 
    WorkStream::run(dof_handler.active_cell_iterators(),
                    *this,
                    &Solid::update_qph_incremental_one_cell,
                    &Solid::copy_local_to_global_UQPH,
                    scratch_data_UQPH,
                    per_task_data_UQPH);
 
    timer.leave_subsection();
  }
 
 
  template <int dim>
  void Solid<dim>::update_qph_incremental_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_UQPH                                     &scratch,
    PerTaskData_UQPH & /*data*/)
  {
    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_q_points, ExcInternalError());
 
    Assert(scratch.solution_grads_u_total.size() == n_q_points,
           ExcInternalError());
    Assert(scratch.solution_values_p_total.size() == n_q_points,
           ExcInternalError());
    Assert(scratch.solution_values_J_total.size() == n_q_points,
           ExcInternalError());
 
    scratch.reset();
 
    scratch.fe_values.reinit(cell);
    scratch.fe_values[u_fe].get_function_gradients(
      scratch.solution_total, scratch.solution_grads_u_total);
    scratch.fe_values[p_fe].get_function_values(
      scratch.solution_total, scratch.solution_values_p_total);
    scratch.fe_values[J_fe].get_function_values(
      scratch.solution_total, scratch.solution_values_J_total);
 
    for (const unsigned int q_point :
         scratch.fe_values.quadrature_point_indices())
    {
      /*Tensor<1,dim> outward_normal;
      //Apply growth at each quadrature point for zeroth newton iteration
        if (dim==2)
        {
        const Point<dim> center(1.0,0.0);  
        outward_normal = 
                scratch.fe_values.quadrature_point(q_point) - center;
        }
        if (dim==3)
        {
        const Point<dim> center(0.0,0.0,0.0);  
        outward_normal = 
                scratch.fe_values.quadrature_point(q_point) - center;
        }
        outward_normal /= outward_normal.norm();*/
      //Tensor<1,dim> outward_normal = scratch.fe_values.quadrature_point(q_point);
      lqph[q_point]->update_values(scratch.solution_grads_u_total[q_point],
                                   scratch.solution_values_p_total[q_point],
                                   scratch.solution_values_J_total[q_point],
                                   //outward_normal,
                                   time.current(),
                                   scratch.growth_update,
                                   parameters.growth_type);
    }
  }
 
 
 
  template <int dim>
  void Solid<dim>::solve_nonlinear_timestep(BlockVector<double> &solution_delta)
  {
    std::cout << std::endl
              << "Timestep " << time.get_timestep() << " @ " << time.current()
              << 's' << std::endl;
 
    BlockVector<double> newton_update(dofs_per_block);
 
    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();
 
    print_conv_header();

    update_qph_incremental(solution_delta,true);
 
    newton_iteration = 0;
    instability = false;
    
    for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
      {
        std::cout << ' ' << std::setw(2) << newton_iteration << ' '
                  << std::flush;
 
        make_constraints(newton_iteration);
        assemble_system();
 
        get_error_residual(error_residual);
        if (newton_iteration == 0)
          error_residual_0 = error_residual;
 
        error_residual_norm = error_residual;
        error_residual_norm.normalize(error_residual_0);
 
        if (newton_iteration > 0 && error_update_norm.u <= parameters.tol_u &&
            error_residual_norm.u <= parameters.tol_f)
          {
            std::cout << " CONVERGED! " << std::endl;
            print_conv_footer();
            break;
          }
        
 
        const std::pair<unsigned int, double> lin_solver_output =
          solve_linear_system(newton_update);
 
        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;
 
        error_update_norm = error_update;
        error_update_norm.normalize(error_update_0);
 
        solution_delta += newton_update;
        update_qph_incremental(solution_delta, false);
 
        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific << lin_solver_output.first << "  "
                  << lin_solver_output.second << "  "
                  << error_residual_norm.norm << "  " << error_residual_norm.u
                  << "  " << error_residual_norm.p << "  "
                  << error_residual_norm.J << "  " << error_update_norm.norm
                  << "  " << error_update_norm.u << "  " << error_update_norm.p
                  << "  " << error_update_norm.J << "  " << std::endl;
      }
 
    AssertThrow(newton_iteration < parameters.max_iterations_NR,
                ExcMessage("No convergence in nonlinear solver!"));
  }
 
 
 
  template <int dim>
  void Solid<dim>::print_conv_header()
  {
    static const unsigned int l_width = 150;
 
    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << '_';
    std::cout << std::endl;
 
    std::cout << "               SOLVER STEP               "
              << " |  LIN_IT   LIN_RES    RES_NORM    "
              << " RES_U     RES_P      RES_J     NU_NORM     "
              << " NU_U       NU_P       NU_J " << std::endl;
 
    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << '_';
    std::cout << std::endl;
  }
 
 
 
  template <int dim>
  void Solid<dim>::print_conv_footer()
  {
    static const unsigned int l_width = 150;
 
    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << '_';
    std::cout << std::endl;
 
    const std::pair<double, double> error_dil = get_error_dilation();
 
    std::cout << "Relative errors:" << std::endl
              << "Displacement:\t" << error_update.u / error_update_0.u
              << std::endl
              << "Force: \t\t" << error_residual.u / error_residual_0.u
              << std::endl
              << "Dilatation:\t" << error_dil.first << std::endl
              << "v / V_0:\t" << error_dil.second * vol_reference << " / "
              << vol_reference << " = " << error_dil.second << std::endl;
  }

  template <int dim>
  double Solid<dim>::compute_vol_current() const
  {
    double vol_current = 0.0;
 
    FEValues<dim> fe_values(fe, qf_cell, update_JxW_values);
 
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        fe_values.reinit(cell);
 
        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
 
        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const double det_F_qp = lqph[q_point]->get_det_F();
            const double JxW      = fe_values.JxW(q_point);
 
            vol_current += det_F_qp * JxW;
          }
      }
    Assert(vol_current > 0.0, ExcInternalError());
    return vol_current;
  }

  template <int dim>
  std::pair<double, double> Solid<dim>::get_error_dilation() const
  {
    double dil_L2_error = 0.0;

    FEValues<dim> fe_values(fe, qf_cell, update_JxW_values);
 
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        fe_values.reinit(cell);
 
        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
 
        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const double det_F_qp   = lqph[q_point]->get_det_F();
            const double J_tilde_qp = lqph[q_point]->get_J_tilde();
            const double the_error_qp_squared =
              Utilities::fixed_power<2>((det_F_qp - J_tilde_qp));
            const double JxW = fe_values.JxW(q_point);
 
            dil_L2_error += the_error_qp_squared * JxW;
          }
      }
 
    return std::make_pair(std::sqrt(dil_L2_error),
                          compute_vol_current() / vol_reference);
  }
 
 
 
  template <int dim>
  void Solid<dim>::get_error_residual(Errors &error_residual)
  {
    BlockVector<double> error_res(dofs_per_block);
 
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_res(i) = system_rhs(i);
 
    error_residual.norm = error_res.l2_norm();
    error_residual.u    = error_res.block(u_dof).l2_norm();
    error_residual.p    = error_res.block(p_dof).l2_norm();
    error_residual.J    = error_res.block(J_dof).l2_norm();
  }
 
 
 
  template <int dim>
  void Solid<dim>::get_error_update(const BlockVector<double> &newton_update,
                                    Errors                    &error_update)
  {
    BlockVector<double> error_ud(dofs_per_block);
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_ud(i) = newton_update(i);
 
    error_update.norm = error_ud.l2_norm();
    error_update.u    = error_ud.block(u_dof).l2_norm();
    error_update.p    = error_ud.block(p_dof).l2_norm();
    error_update.J    = error_ud.block(J_dof).l2_norm();
  }
 
 
 
 
  template <int dim>
  BlockVector<double> Solid<dim>::get_total_solution(
    const BlockVector<double> &solution_delta) const
  {
    BlockVector<double> solution_total(solution_n);
    solution_total += solution_delta;
    return solution_total;
  }
 
  template <int dim>
  void Solid<dim>::assemble_system()
  {
    timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
 
    tangent_matrix = 0.0;
    system_rhs     = 0.0;
 
    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);
 
    PerTaskData_ASM per_task_data(dofs_per_cell);
    ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face);
 
    WorkStream::run(
      dof_handler.active_cell_iterators(),
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
             ScratchData_ASM                                      &scratch,
             PerTaskData_ASM                                      &data) {
        this->assemble_system_one_cell(cell, scratch, data);
      },
      [this](const PerTaskData_ASM &data) {
        this->constraints.distribute_local_to_global(data.cell_matrix,
                                                     data.cell_rhs,
                                                     data.local_dof_indices,
                                                     tangent_matrix,
                                                     system_rhs);
      },
      scratch_data,
      per_task_data);
 
    timer.leave_subsection();
  }
 
  template <int dim>
  void Solid<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM                                      &scratch,
    PerTaskData_ASM                                      &data) const
  {
    data.reset();
    scratch.reset();
    scratch.fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
 
    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_q_points, ExcInternalError());
 
    for (const unsigned int q_point :
         scratch.fe_values.quadrature_point_indices())
      {
        const Tensor<2, dim> F_inv = lqph[q_point]->get_F_inv();
        for (const unsigned int k : scratch.fe_values.dof_indices())
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;
 
            if (k_group == u_dof)
              {
                scratch.grad_Nx[q_point][k] =
                  scratch.fe_values[u_fe].gradient(k, q_point) * F_inv;
                scratch.symm_grad_Nx[q_point][k] =
                  symmetrize(scratch.grad_Nx[q_point][k]);
              }
            else if (k_group == p_dof)
              scratch.Nx[q_point][k] =
                scratch.fe_values[p_fe].value(k, q_point);
            else if (k_group == J_dof)
              scratch.Nx[q_point][k] =
                scratch.fe_values[J_fe].value(k, q_point);
            else
              Assert(k_group <= J_dof, ExcInternalError());
          }
      }
 
    for (const unsigned int q_point :
         scratch.fe_values.quadrature_point_indices())
      {
        const SymmetricTensor<2, dim> tau     = lqph[q_point]->get_tau();
        const Tensor<2, dim>          tau_ns  = lqph[q_point]->get_tau();
        const SymmetricTensor<4, dim> Jc      = lqph[q_point]->get_Jc();
        const double                  det_F   = lqph[q_point]->get_det_F();
        const double                  p_tilde = lqph[q_point]->get_p_tilde();
        const double                  J_tilde = lqph[q_point]->get_J_tilde();
        const double dPsi_vol_dJ   = lqph[q_point]->get_dPsi_vol_dJ();
        const double d2Psi_vol_dJ2 = lqph[q_point]->get_d2Psi_vol_dJ2();
        const SymmetricTensor<2, dim> &I =
          Physics::Elasticity::StandardTensors<dim>::I;
 
        SymmetricTensor<2, dim> symm_grad_Nx_i_x_Jc;
        Tensor<1, dim>          grad_Nx_i_comp_i_x_tau;
 
        const std::vector<double>                  &N = scratch.Nx[q_point];
        const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx =
          scratch.symm_grad_Nx[q_point];
        const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
        const double                       JxW = scratch.fe_values.JxW(q_point);
 
        for (const unsigned int i : scratch.fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;
            const unsigned int i_group = fe.system_to_base_index(i).first.first;
 
            if (i_group == u_dof)
              data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
            else if (i_group == p_dof)
              data.cell_rhs(i) -= N[i] * (det_F - J_tilde) * JxW;
            else if (i_group == J_dof)
              data.cell_rhs(i) -= N[i] * (dPsi_vol_dJ - p_tilde) * JxW;
            else
              Assert(i_group <= J_dof, ExcInternalError());
 
            if (i_group == u_dof)
              {
                symm_grad_Nx_i_x_Jc    = symm_grad_Nx[i] * Jc;
                grad_Nx_i_comp_i_x_tau = grad_Nx[i][component_i] * tau_ns;
              }
 
            for (const unsigned int j :
                 scratch.fe_values.dof_indices_ending_at(i))
              {
                const unsigned int component_j =
                  fe.system_to_component_index(j).first;
                const unsigned int j_group =
                  fe.system_to_base_index(j).first.first;
 
                if ((i_group == j_group) && (i_group == u_dof))
                  {
                    data.cell_matrix(i, j) += symm_grad_Nx_i_x_Jc *  
                                              symm_grad_Nx[j] * JxW; 
 
                    if (component_i == component_j)
                      data.cell_matrix(i, j) +=
                        grad_Nx_i_comp_i_x_tau * grad_Nx[j][component_j] * JxW;
                  }
                else if ((i_group == p_dof) && (j_group == u_dof))
                  {
                    data.cell_matrix(i, j) += N[i] * det_F *               
                                              (symm_grad_Nx[j] * I) * JxW; 
                  }
                else if ((i_group == J_dof) && (j_group == p_dof))
                  data.cell_matrix(i, j) -= N[i] * N[j] * JxW;
                else if ((i_group == j_group) && (i_group == J_dof))
                  data.cell_matrix(i, j) += N[i] * d2Psi_vol_dJ2 * N[j] * JxW;
                else
                  Assert((i_group <= J_dof) && (j_group <= J_dof),
                         ExcInternalError());
              }
          }
      }
 
    for (const unsigned int i : scratch.fe_values.dof_indices())
      for (const unsigned int j :
           scratch.fe_values.dof_indices_starting_at(i + 1))
        data.cell_matrix(i, j) = data.cell_matrix(j, i);
  }

  template <int dim>
  void Solid<dim>::make_constraints(const int it_nr)
  {
    if (it_nr > 1)
    {
      std::cout << " --- " << std::flush;
      return;
    }
    std::cout << " CST " << std::flush;

      if (dim==2)
      {
          constraints.clear();
          const bool apply_dirichlet_bc = (it_nr == 0);
          const FEValuesExtractors::Vector displacement(0);
          const FEValuesExtractors::Scalar y_displacement(1);
          const FEValuesExtractors::Scalar x_displacement(0);
          
          {
              const int boundary_id = 2;
              
              //if (apply_dirichlet_bc == true)
              {
                  VectorTools::interpolate_boundary_values(dof_handler,
                                                           boundary_id,
                                                           Functions::ZeroFunction<dim>(n_components),
                                                           constraints,
                                                           fe.component_mask(x_displacement));
                  
              }
          }
          {
              const int boundary_id = 3;
              
              //if (apply_dirichlet_bc == true)
              {
                  VectorTools::interpolate_boundary_values(dof_handler,
                                                           boundary_id,
                                                           Functions::ZeroFunction<dim>(n_components),
                                                           constraints,
                                                           fe.component_mask(y_displacement));
                  
              }
          }
          {
              const int boundary_id = 0;
              //if (apply_dirichlet_bc == true)
              {
                  VectorTools::interpolate_boundary_values(dof_handler,
                                                           boundary_id,
                                                           Functions::ZeroFunction<dim>(n_components),
                                                           constraints,
                                                           fe.component_mask(displacement));
              }
          }
      }
    
      else if (dim==3)
      {
          constraints.clear();
          const bool apply_dirichlet_bc = (it_nr == 0);
          const FEValuesExtractors::Vector displacement(0);
          const FEValuesExtractors::Scalar x_displacement(0);
          const FEValuesExtractors::Scalar y_displacement(1);
          const FEValuesExtractors::Scalar z_displacement(2);
          
          {
                  const int boundary_id = 2;
                  
                  //if (apply_dirichlet_bc == true)
                  {
                      VectorTools::interpolate_boundary_values(dof_handler,
                                                               boundary_id,
                                                               Functions::ZeroFunction<dim>(n_components),
                                                               constraints,
                                                               fe.component_mask(y_displacement));
                      
                  }
          }
              
              {
                  const int boundary_id = 0;
                  //if (apply_dirichlet_bc == true)
                  {
                      VectorTools::interpolate_boundary_values(dof_handler,
                                                               boundary_id,
                                                               Functions::ZeroFunction<dim>(n_components),
                                                               constraints,
                                                               fe.component_mask(displacement));
                  }
              }
              
      }

    constraints.close();
  }

 
  template <int dim>
  void Solid<dim>::assemble_sc()
  {
    timer.enter_subsection("Perform static condensation");
    std::cout << " ASM_SC " << std::flush;
 
    PerTaskData_SC per_task_data(dofs_per_cell,
                                 element_indices_u.size(),
                                 element_indices_p.size(),
                                 element_indices_J.size());
    ScratchData_SC scratch_data;
 
    WorkStream::run(dof_handler.active_cell_iterators(),
                    *this,
                    &Solid::assemble_sc_one_cell,
                    &Solid::copy_local_to_global_sc,
                    scratch_data,
                    per_task_data);
 
    timer.leave_subsection();
  }
 
 
  template <int dim>
  void Solid<dim>::copy_local_to_global_sc(const PerTaskData_SC &data)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        tangent_matrix.add(data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.cell_matrix(i, j));
  }
 
 
  template <int dim>
  void Solid<dim>::assemble_sc_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_SC                                       &scratch,
    PerTaskData_SC                                       &data)
  {
    data.reset();
    scratch.reset();
    cell->get_dof_indices(data.local_dof_indices);
 
 
    data.k_orig.extract_submatrix_from(tangent_matrix,
                                       data.local_dof_indices,
                                       data.local_dof_indices);
    data.k_pu.extract_submatrix_from(data.k_orig,
                                     element_indices_p,
                                     element_indices_u);
    data.k_pJ.extract_submatrix_from(data.k_orig,
                                     element_indices_p,
                                     element_indices_J);
    data.k_JJ.extract_submatrix_from(data.k_orig,
                                     element_indices_J,
                                     element_indices_J);
 
    data.k_pJ_inv.invert(data.k_pJ);
 
    data.k_pJ_inv.mmult(data.A, data.k_pu);
    data.k_JJ.mmult(data.B, data.A);
    data.k_pJ_inv.Tmmult(data.C, data.B);
    data.k_pu.Tmmult(data.k_bbar, data.C);
    data.k_bbar.scatter_matrix_to(element_indices_u,
                                  element_indices_u,
                                  data.cell_matrix);
 
    data.k_pJ_inv.add(-1.0, data.k_pJ);
    data.k_pJ_inv.scatter_matrix_to(element_indices_p,
                                    element_indices_J,
                                    data.cell_matrix);
  }
 
  template <int dim>
  std::pair<unsigned int, double>
  Solid<dim>::solve_linear_system(BlockVector<double> &newton_update)
  {
    unsigned int lin_it  = 0;
    double       lin_res = 0.0;
 
    if (parameters.use_static_condensation == true)
      {
 
        BlockVector<double> A(dofs_per_block);
        BlockVector<double> B(dofs_per_block);
 
 
        {
          assemble_sc();
 
          tangent_matrix.block(p_dof, J_dof)
            .vmult(A.block(J_dof), system_rhs.block(p_dof));
          tangent_matrix.block(J_dof, J_dof)
            .vmult(B.block(J_dof), A.block(J_dof));
          A.block(J_dof) = system_rhs.block(J_dof);
          A.block(J_dof) -= B.block(J_dof);
          tangent_matrix.block(p_dof, J_dof)
            .Tvmult(A.block(p_dof), A.block(J_dof));
          tangent_matrix.block(u_dof, p_dof)
            .vmult(A.block(u_dof), A.block(p_dof));
          system_rhs.block(u_dof) -= A.block(u_dof);
 
          timer.enter_subsection("Linear solver");
          std::cout << " SLV " << std::flush;
          if (parameters.type_lin == "CG")
            {
              const auto solver_its = static_cast<unsigned int>(
                tangent_matrix.block(u_dof, u_dof).m() *
                parameters.max_iterations_lin);
              const double tol_sol =
                parameters.tol_lin * system_rhs.block(u_dof).l2_norm();
 
              SolverControl solver_control(solver_its, tol_sol);
 
              GrowingVectorMemory<Vector<double>> GVM;
              SolverCG<Vector<double>> solver_CG(solver_control, GVM);
 
              PreconditionSelector<SparseMatrix<double>, Vector<double>>
                preconditioner(parameters.preconditioner_type,
                               parameters.preconditioner_relaxation);
              preconditioner.use_matrix(tangent_matrix.block(u_dof, u_dof));
 
              solver_CG.solve(tangent_matrix.block(u_dof, u_dof),
                              newton_update.block(u_dof),
                              system_rhs.block(u_dof),
                              preconditioner);
 
              lin_it  = solver_control.last_step();
              lin_res = solver_control.last_value();
            }
          else if (parameters.type_lin == "Direct")
            {
              SparseDirectUMFPACK A_direct;
              A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
              A_direct.vmult(newton_update.block(u_dof),
                             system_rhs.block(u_dof));
 
              lin_it  = 1;
              lin_res = 0.0;
            }
          else
            Assert(false, ExcMessage("Linear solver type not implemented"));
 
          timer.leave_subsection();
        }
 
        constraints.distribute(newton_update);
 
        timer.enter_subsection("Linear solver postprocessing");
        std::cout << " PP " << std::flush;
 
        {
          tangent_matrix.block(p_dof, u_dof)
            .vmult(A.block(p_dof), newton_update.block(u_dof));
          A.block(p_dof) *= -1.0;
          A.block(p_dof) += system_rhs.block(p_dof);
          tangent_matrix.block(p_dof, J_dof)
            .vmult(newton_update.block(J_dof), A.block(p_dof));
        }
 
        constraints.distribute(newton_update);
 
        {
          tangent_matrix.block(J_dof, J_dof)
            .vmult(A.block(J_dof), newton_update.block(J_dof));
          A.block(J_dof) *= -1.0;
          A.block(J_dof) += system_rhs.block(J_dof);
          tangent_matrix.block(p_dof, J_dof)
            .Tvmult(newton_update.block(p_dof), A.block(J_dof));
        }
 
        constraints.distribute(newton_update);
 
        timer.leave_subsection();
      }
    else
      {
        std::cout << " ------ " << std::flush;
 
        timer.enter_subsection("Linear solver");
        std::cout << " SLV " << std::flush;
 
        if (parameters.type_lin == "CG")
          {
 
            const Vector<double> &f_u = system_rhs.block(u_dof);
            const Vector<double> &f_p = system_rhs.block(p_dof);
            const Vector<double> &f_J = system_rhs.block(J_dof);
 
            Vector<double> &d_u = newton_update.block(u_dof);
            Vector<double> &d_p = newton_update.block(p_dof);
            Vector<double> &d_J = newton_update.block(J_dof);
 
            const auto K_uu =
              linear_operator(tangent_matrix.block(u_dof, u_dof));
            const auto K_up =
              linear_operator(tangent_matrix.block(u_dof, p_dof));
            const auto K_pu =
              linear_operator(tangent_matrix.block(p_dof, u_dof));
            const auto K_Jp =
              linear_operator(tangent_matrix.block(J_dof, p_dof));
            const auto K_JJ =
              linear_operator(tangent_matrix.block(J_dof, J_dof));
 
            PreconditionSelector<SparseMatrix<double>, Vector<double>>
              preconditioner_K_Jp_inv("jacobi");
            preconditioner_K_Jp_inv.use_matrix(
              tangent_matrix.block(J_dof, p_dof));
            ReductionControl solver_control_K_Jp_inv(
              static_cast<unsigned int>(tangent_matrix.block(J_dof, p_dof).m() *
                                        parameters.max_iterations_lin),
              1.0e-30,
              parameters.tol_lin);
            SolverSelector<Vector<double>> solver_K_Jp_inv;
            solver_K_Jp_inv.select("cg");
            solver_K_Jp_inv.set_control(solver_control_K_Jp_inv);
            const auto K_Jp_inv =
              inverse_operator(K_Jp, solver_K_Jp_inv, preconditioner_K_Jp_inv);
 
            const auto K_pJ_inv     = transpose_operator(K_Jp_inv);
            const auto K_pp_bar     = K_Jp_inv * K_JJ * K_pJ_inv;
            const auto K_uu_bar_bar = K_up * K_pp_bar * K_pu;
            const auto K_uu_con     = K_uu + K_uu_bar_bar;
 
            PreconditionSelector<SparseMatrix<double>, Vector<double>>
              preconditioner_K_con_inv(parameters.preconditioner_type,
                                       parameters.preconditioner_relaxation);
            preconditioner_K_con_inv.use_matrix(
              tangent_matrix.block(u_dof, u_dof));
            ReductionControl solver_control_K_con_inv(
              static_cast<unsigned int>(tangent_matrix.block(u_dof, u_dof).m() *
                                        parameters.max_iterations_lin),
              1.0e-30,
              parameters.tol_lin);
            SolverSelector<Vector<double>> solver_K_con_inv;
            solver_K_con_inv.select("cg");
            solver_K_con_inv.set_control(solver_control_K_con_inv);
            const auto K_uu_con_inv =
              inverse_operator(K_uu_con,
                               solver_K_con_inv,
                               preconditioner_K_con_inv);
 
            d_u =
              K_uu_con_inv * (f_u - K_up * (K_Jp_inv * f_J - K_pp_bar * f_p));
 
            timer.leave_subsection();
 
            timer.enter_subsection("Linear solver postprocessing");
            std::cout << " PP " << std::flush;
 
            d_J = K_pJ_inv * (f_p - K_pu * d_u);
            d_p = K_Jp_inv * (f_J - K_JJ * d_J);
 
            lin_it  = solver_control_K_con_inv.last_step();
            lin_res = solver_control_K_con_inv.last_value();
          }
        else if (parameters.type_lin == "Direct")
          {
            SparseDirectUMFPACK A_direct;
            A_direct.initialize(tangent_matrix);
            A_direct.vmult(newton_update, system_rhs);
 
            lin_it  = 1;
            lin_res = 0.0;
 
            std::cout << " -- " << std::flush;
          }
        else
          Assert(false, ExcMessage("Linear solver type not implemented"));
 
        timer.leave_subsection();
 
        constraints.distribute(newton_update);
      }
 
    return std::make_pair(lin_it, lin_res);
  }
 
  template <int dim>
  void Solid<dim>::output_results() const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
 
    std::vector<std::string> solution_name(dim, "displacement");
    solution_name.emplace_back("pressure");
    solution_name.emplace_back("dilatation");
 
    DataOutBase::VtkFlags output_flags;
    output_flags.write_higher_order_cells       = true;
    output_flags.physical_units["displacement"] = "m";
    data_out.set_flags(output_flags);
    /*std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_interpretation(
          1, DataComponentInterpretation::component_is_scalar);*/
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_n,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    //data_out.add_data_vector(new_material_ids, "MaterialID", DataOut<dim>::type_cell_data, data_interpretation);

    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
      soln(i) = solution_n(i);
    const MappingQEulerian<dim> q_mapping(degree, dof_handler, soln);
    data_out.build_patches(q_mapping, degree);
 
    std::ofstream output("solution-" + std::to_string(dim) + "d-" +
                         std::to_string(time.get_timestep()) + ".vtk");
    data_out.write_vtk(output);
  }
 
} // namespace Step44
 
 
int main(int argc, char *argv[])
{
  using namespace Step44;
  //std::freopen("convergence.txt","w",stdout);
    try
    {
        Solid<2>         solid("parameters.prm");
        solid.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}
