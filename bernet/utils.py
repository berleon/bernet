# Copyright 2015 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib

import urllib.request


def download(url: str, file) -> bool:
    # TODO: make the download bar nicer (e.g MBytes, a real bar, ...)
    u = urllib.request.urlopen(url)
    meta = u.info()
    file_size = int(meta["Content-Length"])
    print("Downloading: %s Bytes: %s" % (url, file_size))

    file_size_dl = 0
    block_sz = 2**16
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        file.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl,
                                       file_size_dl * 100. / file_size)
        print(status, end='\r')

    file.flush()


def sha256_file(file, block_size: int=65536) -> str:
    """Checks if the file has the same sha256-hash as given by `sha256sum`"""
    sha = hashlib.sha256()

    file.seek(0)
    buf = file.read(block_size)
    while len(buf) > 0:
        sha.update(buf)
        buf = file.read(block_size)

    return sha.hexdigest()

