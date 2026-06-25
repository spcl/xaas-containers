import logging
from functools import reduce

from xaas.config import BuildSystemArguments, DerivedDockerImageDescriptor
from xaas.actions.action import Action
from xaas.actions.build import BuildGenerator
from xaas.actions.build import Config as BuildConfig
from xaas.config import DeployConfig
from xaas.docker import DockerBuildFeatures
from xaas.util.dockerfile import DockerfileBuilder, DockerfileStage, CopyStep, RunStep


class Deployment(Action):
    def __init__(self, parallel_workers: int):
        super().__init__(
            name="dockerimagebuilder",
            description="Create a Docker image containing all build directories for IR analysis.",
        )
        self.parallel_workers = parallel_workers

    def execute(self, config: DeployConfig) -> bool:
        name = BuildGenerator.generate_name(config.features_boolean, config.features_select)

        docker_builder = self.docker_runner.try_get_buildkit_builder() or self.docker_runner

        dockerfile_content = self._generate_dockerfile(name, config, docker_builder.get_build_features())

        image_name = config.ir_image.split(":")[1].removesuffix("-ir")
        image_name = f"{config.docker_repository}:{image_name}-deploy-{name}"

        docker_builder.build(path=None, dockerfile_content=dockerfile_content, tag=image_name, show_progress=True)

        logging.info(f"[{self.name}] Successfully built Docker image {image_name}")

        return True

    def _generate_dockerfile(self, build_dir_name: str, deploy_config: DeployConfig, build_features: DockerBuildFeatures) -> str:
        # extract the original build config from the ir container image
        ir_image = self.docker_runner.get_image(deploy_config.ir_image)
        build_config = BuildConfig.from_yaml(ir_image.labels["xaas.BuildConfig"])

        # determine the effective build arguments for the deployment config
        effective_cpu_architecture = deploy_config.cpu_architecture
        effective_run_config = build_config.for_target(effective_cpu_architecture)
        effective_base_builder_image, effective_base_runtime_image = effective_run_config.effective_docker_images()

        states_boolean = deploy_config.features_boolean
        states_select = deploy_config.features_select

        arguments = reduce(BuildSystemArguments.merge, [
            # universal build arguments
            effective_run_config.build_args,

            # include build arguments for the current feature selection
            *[ effective_run_config.features_boolean[feat].args_for_state(state) for feat, state in states_boolean.items() ],
            *[ effective_run_config.features_select[feat][state] for feat, state in states_select.items() ],
        ])

        # figure out which layers we need to include in the builder and runtime images
        prepared_dependencies = [ d.prepare(effective_cpu_architecture, states_boolean, states_select) for d in arguments.dependencies ]

        builder_image_desc, runtime_image_desc = DerivedDockerImageDescriptor.create_builder_and_runtime(effective_base_builder_image, effective_base_runtime_image, prepared_dependencies)

        # generate a dockerfile for deploying the final image
        dockerfile_builder = DockerfileBuilder()

        # dockerfile part 1: make the source, IR and build directories accessible to the builder image, then run all the build commands

        build_command = f"cd /build && parallel --eta --halt now,fail=1 -j{self.parallel_workers} < build.sh && make -j{self.parallel_workers}"

        if build_features.dockerfile_supports_run_mount:
            # if BuildKit is supported, we can avoid having to copy all the dependencies and source/IR/build dirs into the base image by
            # bind-mounting them on top of the base builder image
            compile_stage = dockerfile_builder.add_stage(builder_image_desc.run_in_prepared_context(dockerfile_builder, RunStep(
                mounts=[
                    RunStep.BindMount(
                        from_context=deploy_config.ir_image,
                        source="/irs",
                        target="/irs",
                        rw=True,
                    ),
                    RunStep.BindMount(
                        from_context=deploy_config.ir_image,
                        source="/source",
                        target="/source",
                    ),
                    RunStep.BindMount(
                        from_context=deploy_config.ir_image,
                        source=f"/builds/build_{build_dir_name}",
                        target=f"/builds/build_{build_dir_name}",
                    ),
                ],
                command=f"cp -a --reflink=auto /builds/build_{build_dir_name} /build && {build_command}",
            )), "compile")
        else:
            # if BuildKit is supported, we'll have to fall back to the less efficient approach of explicitly copying the dependencies
            # and source/IR/build dirs on top of the builder image
            builder_stage = dockerfile_builder.add_stage(
                builder_image_desc.prepared_dockerfile_stage(), "builder")

            compile_stage = dockerfile_builder.add_stage(DockerfileStage(
                from_context=builder_stage,
                steps=[
                    # unfortunately, we can't copy /irs and /source in a single step (would be possible if we could use --parents,
                    # but we don't have BuildKit...)
                    CopyStep(
                        from_context=deploy_config.ir_image,
                        source="/source",
                        target="/source",
                        # we want to use --link here: deploying for multiple configurations may result in a different builder_stage
                        # if the dependencies change, using --link will allow this copy step to be simply rebased
                        link=build_features.dockerfile_supports_copy_link,
                    ),
                    CopyStep(
                        from_context=deploy_config.ir_image,
                        source="/irs",
                        target="/irs",
                        # we want to use --link here: deploying for multiple configurations may result in a different builder_stage
                        # if the dependencies change, using --link will allow this copy step to be simply rebased
                        link=build_features.dockerfile_supports_copy_link,
                    ),
                    CopyStep(
                        from_context=deploy_config.ir_image,
                        source=f"/builds/build_{build_dir_name}",
                        target=f"/build",
                    ),
                    RunStep(command=build_command)
                ],
            ), "compile")

        # dockerfile part 2: prepare the runtime image, then copy the build output into it

        runner_stage = dockerfile_builder.add_stage(
            runtime_image_desc.prepared_dockerfile_stage(), "runner")

        terminal_stage = DockerfileStage(
            from_context=runner_stage,
            steps=[CopyStep(
                from_context=compile_stage,
                source="/build",
                target="/build",
                link=build_features.dockerfile_supports_copy_link,
            )]
        )

        dockerfile = dockerfile_builder.build(terminal_stage)
        return dockerfile.to_str()

    def validate(self, config: DeployConfig) -> bool:
        return True
