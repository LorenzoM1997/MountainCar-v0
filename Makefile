docker: Dockerfile
	docker build -t mountain-car:latest -f Dockerfile .

exec:
	docker run -it --rm \
	-v ${CURDIR}:/w \
	-w /w \
	mountain-car:latest
