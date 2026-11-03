# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# Modules
import itertools
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

int_list = ["rk2"]
# if this is updated make sure to update the assert statements for the number of MPI ranks, too
res_list = [16, 32, 64]  # resolution for linear convergence
all_cfgs = list(itertools.product(int_list, res_list))


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        """
        Any preprocessing that is needed before the drive is run can be done in
        this method

        This includes preparing files or any other pre processing steps that
        need to be implemented.  The method also provides access to the
        parameters object which controls which parameters are being used to run
        the driver.
        It is possible to append arguments to the driver_cmd_line_args if it is
        desired to  override the parthenon input file. Each element in the list
        is simply a string of the form '<block>/<field>=<value>', where the
        contents of the string are exactly what one would type on the command
        line run running a parthenon driver.
        As an example if the following block was uncommented it would overwrite
        any of the parameters that were specified in the parthenon input file
        parameters.driver_cmd_line_args = ['output1/file_type=vtk',
                'output1/variable=cons',
                'output1/dt=0.4',
                'time/tlim=0.4',
                'mesh/nx1=400']
        """
        # make sure we can evenly distribute the MeshBlock sizes
        err_msg = "Num ranks must be multiples of 2 for convergence test."
        assert parameters.num_ranks == 1 or parameters.num_ranks % 2 == 0, err_msg
        # ensure a minimum block size of 4
        assert (
            res_list[0] / parameters.num_ranks >= 4
        ), "Use <= 8 ranks for convergence test."
        integrator, res = all_cfgs[step - 1]  # step counting starts at 1
        mb_nx1 = (2 * res) // parameters.num_ranks
        # ensure that nx1 is <= 256 when using scratch (A100 limit on test system)
        while mb_nx1 > 256:
            mb_nx1 //= 2
        parameters.driver_cmd_line_args = [
            "parthenon/mesh/nx1=%d" % (2 * res),
            "parthenon/meshblock/nx1=%d" % mb_nx1,
            "parthenon/mesh/nx2=%d" % res,
            "parthenon/meshblock/nx2=%d" % res,
            "parthenon/mesh/nx3=%d" % res,
            "parthenon/meshblock/nx3=%d" % res,
            "parthenon/time/integrator=%s" % integrator,
            "parthenon/time/nlim=300", # safety measure to not run to infty
        ]

        return parameters
    def Analyse(self, parameters):
        """
        Analyze the output and determine if the test passes.

        This function is called after the driver has been executed. It is
        responsible for reading whatever data it needs and making a judgment
        about whether or not the test passes. It takes no inputs. Output should
        be True (test passes) or False (test fails).
        The parameters that are passed in provide the paths to relevant
        locations and commands. Of particular importance is the path to the
        output folder. All files from a drivers run should appear in and output
        folder located in
        parthenon/tst/regression/test_suites/test_name/output.

        It is possible in this function to read any of the output files such as
        hdf5 output and compare them to expected quantities.

        """
        error_file = os.path.join(parameters.output_path, "linearwave-errors.dat")
        try:
            with open(error_file, "r") as f:
                lines = f.readlines()
        except OSError:
            print("linearwave-errors.dat file not accessible")
            return False

        if len(lines) != len(all_cfgs) + 1:
            print(
                f"Unexpected number of lines in {error_file}: "
                f"expected {len(all_cfgs) + 1}, got {len(lines)}"
            )
            return False

        try:
            data = np.atleast_2d(np.genfromtxt(error_file))
        except (OSError, ValueError) as exc:
            print(f"Failed to read {error_file}: {exc}")
            return False

        if data.shape[0] != len(all_cfgs) or data.shape[1] < 5:
            print(f"Unexpected shape for linear wave errors: {data.shape}")
            return False

        # Column 0 is nx1 (= 2 * res); column 4 is the L1 error.
        expected_resolutions = np.tile(2 * np.asarray(res_list), len(int_list))
        if not np.array_equal(data[:, 0], expected_resolutions):
            print(
                "Unexpected resolutions in linearwave-errors.dat: "
                f"expected {expected_resolutions}, got {data[:, 0]}"
            )
            return False
        if not np.all(np.isfinite(data[:, [0, 4]])) or np.any(data[:, 4] <= 0):
            print("Invalid (non-finite or non-positive) linear wave errors")
            return False

        # Allow modest pre-asymptotic deviations from the expected second order.
        min_order = 1.6
        analyze_status = True
        n_res = len(res_list)
        markers = "ov^<>sp*hXD"
        for i, integrator in enumerate(int_list):
            group = data[i * n_res : (i + 1) * n_res]
            resolutions, errors = group[:, 0], group[:, 4]
            orders = np.log(errors[:-1] / errors[1:]) / np.log(
                resolutions[1:] / resolutions[:-1]
            )
            for j, order in enumerate(orders):
                print(
                    f"{integrator}: nx1={int(resolutions[j])} -> "
                    f"{int(resolutions[j + 1])}: observed order = {order:.3f} "
                    f"(minimum {min_order:.1f})"
                )
                if not np.isfinite(order) or order < min_order:
                    print(f"Convergence order check FAILED for {integrator}")
                    analyze_status = False

            plt.plot(resolutions, errors, marker=markers[i], label=integrator)

        # Retain the original absolute-error regression guard.
        if data[-1, 4] > 4.63e-08:
            print("Finest-resolution L1 error check FAILED")
            analyze_status = False

        plt.plot([32, 128], [4e-7, 4e-7 / (128 / 32) ** 2], "--", label="second order")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("L1 err")
        plt.xlabel("Linear resolution")
        plt.savefig(
            os.path.join(parameters.output_path, "linearwave-errors.png"),
            bbox_inches="tight",
        )
        plt.close()

        return analyze_status
