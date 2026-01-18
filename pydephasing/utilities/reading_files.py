#
#   utility functions for reading files
#

#
#   detect binary or text file
#

def check_binary_file(filename, nbytes=4096):
    """
    Returns True if file is binary, False if text.
    """
    with open(filename, "rb") as f:
        chunk = f.read(nbytes)
    return b"\x00" in chunk

#
#   build line offsets
#

def build_line_offsets(filename):
	"""Return byte offsets for each line (0-based, INCLUDING header)."""
	offsets = []
	with open(filename, "rb") as f:
		offsets.append(f.tell())
		for line in f:
			offsets.append(f.tell())
	return np.asarray(offsets, dtype=np.int64)