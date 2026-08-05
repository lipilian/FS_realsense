#include <StApi_TL.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kRequiredCameraCount = 2;

} // namespace

int main() {
    try {
        if (std::getenv("GENICAM_GENTL64_PATH") == nullptr &&
            setenv("GENICAM_GENTL64_PATH", FFS_SENTECH_GENTL_DIRECTORY, 0) != 0) {
            throw std::runtime_error("Unable to configure GENICAM_GENTL64_PATH");
        }
        // This initializes and tears down the StApi runtime automatically.
        StApi::CStApiAutoInit stapi;
        StApi::CIStSystemPtr system(StApi::CreateIStSystem());
        StApi::CIStDevicePtrArray devices;

        std::cout << "Looking for " << kRequiredCameraCount << " Sentech cameras...\n";
        for (std::size_t index = 0; index < kRequiredCameraCount; ++index) {
            try {
                // CreateFirstIStDevice opens the first currently unclaimed
                // device, so calling it twice verifies two independent links.
                StApi::IStDeviceReleasable *device = system->CreateFirstIStDevice();
                devices.Register(device);

                const StApi::IStDeviceInfo *info = device->GetIStDeviceInfo();
                std::cout << "Camera " << (index + 1) << ": connected"
                          << " | name: " << info->GetDisplayName()
                          << " | serial: " << info->GetSerialNumber() << '\n';
            } catch (const GenICam::GenericException &error) {
                std::cerr << "Camera " << (index + 1)
                          << ": connection failed. " << error.GetDescription() << '\n';
                std::cerr << "FAIL: two Sentech cameras could not be connected.\n";
                return 2;
            }
        }

        std::cout << "PASS: both Sentech cameras are connected.\n";
        // No acquisition is started: this executable is only a connection test.
        return 0;
    } catch (const GenICam::GenericException &error) {
        std::cerr << "Sentech StApi initialization failed: " << error.GetDescription() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Unexpected error: " << error.what() << '\n';
    }

    return 1;
}
