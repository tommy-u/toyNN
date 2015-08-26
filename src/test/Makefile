CC = gcc
WFLAGS = -Wall -Werror
LDLIBS = -lm -I ../lib
TESTS = not xor xor_data_file
DEBUGFLAGS = -g
#OPTIMIZEFLAGS = -O2
all: tests

%: %.c
	$(CC) $@.c $(WFLAGS) $(LDLIBS) $(DEBUGFLAGS) -o $@

tests: $(TESTS)

clean:
	rm -r $(TESTS)
