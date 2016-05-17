#ifndef thrust_operators_H
#define thrust_operators_H

template<typename Element>
__host__ __device__ Element max_pixel(Element pixel1, Element pixel2)
{
 return pixel1 > pixel2 ? pixel1 : pixel2;
}

template<typename Element>
struct SquareOperation
{
    __host__ __device__ Element operator()(const Element& element) const
    {
        return element * element;
    }
};

template<typename Element>
struct SquareRootOperation
{
    __host__ __device__ Element operator()(const Element& element) const
    {
        return std::sqrt(element);
    }
};

template<typename Element>
struct MultiplyByConstant
{
    const Element constant_factor;

    __host__ __device__
    MultiplyByConstant(const Element constant_factor) : constant_factor(constant_factor) {}

    __host__ __device__
    Element operator()(const Element& element) const
    {
        return element * constant_factor;
    }
};

template<typename Element>
struct MultiplyByConstantAndAddOperation  // Saxpy-Operation
{
    const Element constant_factor;

    __host__ __device__
    MultiplyByConstantAndAddOperation(const Element constant_factor) : constant_factor(constant_factor) {}

    __host__ __device__
    Element operator()(const Element& element1, const Element& element2) const
    {
        return element1 * constant_factor + element2;
    }
};

template<typename Element>
struct ProjectNormalizedGradientMagnitude1 : public thrust::binary_function<Element, Element, Element>
{
    __host__ __device__
    Element operator()(Element& element1, Element& element2) const
    {
        const Element magnitude = std::sqrt(element1*element1 + element2*element2);
        const Element normalization = max_pixel<Element>(magnitude, 1.0);
        return element1 / normalization;
    }
};

template<typename Element>
struct ProjectNormalizedGradientMagnitude2 : public thrust::binary_function<Element, Element, Element>
{
    __host__ __device__
    Element operator()(Element& element1, Element& element2) const
    {
        const Element magnitude = std::sqrt(element1*element1 + element2*element2);
        const Element normalization = max_pixel<Element>(magnitude, 1.0f);
        return element2 / normalization;
    }
};

template<typename Element>
struct GradientMagnitude : public thrust::binary_function<Element, Element, Element>
{
    __host__ __device__
    Element operator()(Element& element1, Element& element2) const
    {
        return std::sqrt(element1*element1 + element2*element2);
    }
};

template<typename Element>
struct GradientMagnitudeSquare : public thrust::binary_function<Element, Element, Element>
{
    __host__ __device__
    Element operator()(Element& element1, Element& element2) const
    {
        return element1*element1 + element2*element2;
    }
};

template<typename Element>
struct InverseMinus : public thrust::binary_function<Element, Element, Element>
{
    __host__ __device__
    Element operator()(Element& element1, Element& element2) const
    {
        return element2 - element1;
    }
};

// FROM: https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

#endif // thrust_operators_H
